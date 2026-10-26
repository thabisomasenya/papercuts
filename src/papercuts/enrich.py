"""Corpus enrichment: abstract backfill, citation snowballing, near-duplicate detection."""

from __future__ import annotations

import asyncio
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher

from papercuts.sources.openalex import OpenAlexClient, OpenAlexError
from papercuts.sources.semantic_scholar import SemanticScholarClient
from papercuts.storage.db import Database

S2_SOURCE = "semantic_scholar"
SNOWBALL_SOURCE = "openalex:snowball"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record_attempt(db: Database, paper_id: str, source: str, found: bool) -> None:
    db.conn.execute(
        "INSERT OR REPLACE INTO enrichment_attempts (paper_id, source, found, attempted_at) VALUES (?, ?, ?, ?)",
        (paper_id, source, int(found), _now()),
    )


# ----------------------------------------------------------------------------- abstract backfill


@dataclass
class BackfillSummary:
    candidates: int
    filled: int
    not_found: int
    no_abstract_there: int
    requests: int


async def backfill_abstracts(
    db: Database, client: SemanticScholarClient, *, batch_size: int = 100, retry_missing: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> BackfillSummary:
    """Fill missing abstracts from Semantic Scholar, matching on DOI. Papers already tried are skipped."""
    skip = "" if retry_missing else (
        "AND p.id NOT IN (SELECT paper_id FROM enrichment_attempts WHERE source = 'semantic_scholar')"
    )
    rows = db.conn.execute(
        f"SELECT p.id, p.doi FROM papers p WHERE (p.abstract IS NULL OR p.abstract = '') "
        f"AND p.doi IS NOT NULL {skip} ORDER BY p.cited_by_count DESC"
    ).fetchall()
    filled = not_found = empty = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        results = await client.batch_by_doi([r["doi"] for r in batch])
        with db.conn:
            for row, res in zip(batch, results):
                abstract = (res or {}).get("abstract") or ""
                abstract = abstract.strip()
                if res is None:
                    not_found += 1
                elif not abstract:
                    empty += 1
                else:
                    db.conn.execute(
                        "UPDATE papers SET abstract = ?, abstract_source = ?, updated_at = ? WHERE id = ?",
                        (abstract, S2_SOURCE, _now(), row["id"]),
                    )
                    filled += 1
                _record_attempt(db, row["id"], S2_SOURCE, bool(abstract))
        if on_progress:
            on_progress(min(start + batch_size, len(rows)), len(rows))
    return BackfillSummary(len(rows), filled, not_found, empty, client.requests_made)


# ----------------------------------------------------------------------------- snowballing


@dataclass
class SnowballCandidate:
    id: str
    cited_by_in_corpus: int


def snowball_candidates(db: Database, min_citing: int = 3, limit: int = 50) -> list[SnowballCandidate]:
    """Papers outside the corpus that at least `min_citing` corpus papers cite, most-cited first."""
    rows = db.conn.execute(
        """
        SELECT c.cited_id, count(*) AS n FROM citations c
        LEFT JOIN papers p ON p.id = c.cited_id
        WHERE p.id IS NULL
          AND c.cited_id NOT IN (SELECT dropped_id FROM merges)
          AND c.cited_id NOT IN (SELECT paper_id FROM enrichment_attempts WHERE source = ? AND found = 0)
        GROUP BY c.cited_id HAVING n >= ?
        ORDER BY n DESC, c.cited_id LIMIT ?
        """,
        (SNOWBALL_SOURCE, min_citing, limit),
    ).fetchall()
    return [SnowballCandidate(r["cited_id"], r["n"]) for r in rows]


@dataclass
class SnowballSummary:
    run_id: int
    candidates: int
    added: int
    failed: list[str] = field(default_factory=list)
    requests: int = 0


async def snowball_backward(
    db: Database, client: OpenAlexClient, *, min_citing: int = 3, limit: int = 50,
    on_progress: Callable[[int, int], None] | None = None,
) -> SnowballSummary:
    """Backward snowballing: fetch the works your corpus cites most. Single-work lookups are free on OpenAlex."""
    cands = snowball_candidates(db, min_citing, limit)
    run_id = db.start_run(f"backward snowball (cited by ≥{min_citing} corpus papers)",
                          {"min_citing": min_citing, "limit": limit}, SNOWBALL_SOURCE, limit)
    added, failed, done = 0, [], 0

    async def fetch(c: SnowballCandidate):
        try:
            return c, await client.get_work(c.id), None
        except OpenAlexError as e:
            return c, None, e

    try:
        for rank, task in enumerate(asyncio.as_completed([fetch(c) for c in cands]), 1):
            cand, paper, err = await task
            done += 1
            if paper is None:
                failed.append(cand.id)
                if err is not None and "404" in str(err):
                    _record_attempt(db, cand.id, SNOWBALL_SOURCE, False)
            else:
                stored_id, is_new = db.upsert_paper(paper)
                if stored_id != cand.id:  # OpenAlex redirected a merged work to its canonical id
                    db._repoint_citations(cand.id, stored_id)
                db.add_run_paper(run_id, stored_id, rank)
                added += int(is_new)
                _record_attempt(db, cand.id, SNOWBALL_SOURCE, True)
            if on_progress:
                on_progress(done, len(cands))
        db.conn.commit()
    except BaseException as e:
        db.conn.commit()
        db.finish_run(run_id, n_retrieved=done - len(failed), n_new=added, total_available=len(cands),
                      status="failed", error=str(e)[:500])
        raise
    db.finish_run(run_id, n_retrieved=len(cands) - len(failed), n_new=added, total_available=len(cands))
    return SnowballSummary(run_id, len(cands), added, failed, client.requests_made)


# ----------------------------------------------------------------------------- near-duplicates

_NOT_DUPLICATES = ("corrigendum", "erratum", "correction", "retraction", "retracted", "reply to",
                   "response to", "comment on", "expression of concern")
_TYPE_RANK = {"article": 0, "review": 1, "book-chapter": 2, "conference-paper": 2, "preprint": 5, "dissertation": 6}


def normalise_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"<[^>]+>", " ", t)  # stray HTML like <i>
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class DuplicatePair:
    keep: dict
    drop: dict
    similarity: float
    reason: str


def _preference(p: dict) -> tuple:
    """Lower sorts first = the record to keep."""
    return (
        _TYPE_RANK.get(p["type"] or "", 3),
        0 if p["doi"] else 1,
        0 if p["abstract"] else 1,
        -(p["cited_by_count"] or 0),
        p["id"],
    )


def find_duplicates(db: Database, threshold: float = 0.93, max_year_gap: int = 2) -> list[DuplicatePair]:
    """Same work stored twice under different ids (typically preprint + journal version).

    Candidates are blocked on the first two title words, compared with a character-level
    similarity ratio, and must be published within `max_year_gap` years of each other.
    """
    rows = [dict(r) for r in db.conn.execute(
        "SELECT id, title, year, type, doi, abstract IS NOT NULL AND abstract != '' AS abstract, "
        "cited_by_count, venue FROM papers"
    )]
    blocks: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        r["_norm"] = normalise_title(r["title"])
        if len(r["_norm"]) < 20:  # "Introduction", "Editorial" … too generic to match on
            continue
        if r["_norm"].startswith(_NOT_DUPLICATES):  # "Corrigendum to <title>" is a different record
            continue
        blocks[" ".join(r["_norm"].split()[:2])].append(r)

    pairs: list[DuplicatePair] = []
    seen: set[str] = set()
    for block in blocks.values():
        for i, a in enumerate(block):
            for b in block[i + 1:]:
                if a["year"] and b["year"] and abs(a["year"] - b["year"]) > max_year_gap:
                    continue
                sim = 1.0 if a["_norm"] == b["_norm"] else SequenceMatcher(None, a["_norm"], b["_norm"]).ratio()
                if sim < threshold:
                    continue
                keep, drop = sorted((a, b), key=_preference)
                if drop["id"] in seen or keep["id"] in seen:
                    continue
                types = {a["type"], b["type"]}
                if "preprint" in types and len(types) > 1:
                    reason = "preprint and published version"
                elif a["doi"] and b["doi"]:
                    reason = "same title, different DOIs"
                else:
                    reason = "same title"
                seen.add(drop["id"])
                pairs.append(DuplicatePair(
                    {k: v for k, v in keep.items() if not k.startswith("_")},
                    {k: v for k, v in drop.items() if not k.startswith("_")},
                    round(sim, 3), reason,
                ))
    return sorted(pairs, key=lambda p: -p.similarity)
