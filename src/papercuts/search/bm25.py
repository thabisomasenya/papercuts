"""BM25 keyword search over title + abstract using SQLite FTS5."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

# Title matches count more than abstract matches. FTS5's bm25() returns *lower is
# better* (negative numbers), so we negate it to get a conventional score.
TITLE_WEIGHT = 3.0
ABSTRACT_WEIGHT = 1.0
HL_OPEN, HL_CLOSE = "⟦", "⟧"  # snippet highlight markers (unlikely to appear in abstracts)

_TOKEN = re.compile(r'"[^"]+"|[\w\-]+', re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it",
    "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with", "how", "what",
    "does", "do", "which", "between", "effect", "effects", "impact",
}


@dataclass
class SearchHit:
    id: str
    title: str
    year: int | None
    venue: str | None
    cited_by_count: int
    score: float
    snippet: str
    ranks: dict[str, int] = field(default_factory=dict)  # method -> rank (hybrid explains itself)


def build_fts_query(query: str, mode: str = "any") -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    - Quoted phrases are kept as phrases: '"carbon fixation" algae'
    - Every other token is quoted, so punctuation and FTS5 keywords (AND, NOT, NEAR)
      in user input can't break the query.
    - Stopwords are dropped (unless that would leave nothing).
    - mode="any" ORs the terms (better recall; BM25 still ranks multi-term matches first),
      mode="all" ANDs them.
    """
    if mode not in {"any", "all"}:
        raise ValueError("mode must be 'any' or 'all'")
    tokens = _TOKEN.findall(query)
    terms: list[str] = []
    for tok in tokens:
        if tok.startswith('"'):
            inner = tok.strip('"').replace('"', "").strip()
            if inner:
                terms.append(f'"{inner}"')
        else:
            word = tok.strip("-").lower()
            if word and word not in _STOPWORDS:
                terms.append(f'"{word}"')
    if not terms:  # query was all stopwords: fall back to using them
        terms = [f'"{t.strip(chr(34)).lower()}"' for t in tokens if t.strip('"-')]
    if not terms:
        return ""
    joiner = " OR " if mode == "any" else " AND "
    return joiner.join(dict.fromkeys(terms))  # de-duplicate, keep order


def bm25_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
    mode: str = "any",
    raw: bool = False,
    from_year: int | None = None,
    to_year: int | None = None,
    min_citations: int | None = None,
) -> list[SearchHit]:
    match = query if raw else build_fts_query(query, mode)
    if not match:
        return []
    where = ["papers_fts MATCH ?"]
    params: list[object] = [match]
    if from_year is not None:
        where.append("p.year >= ?")
        params.append(from_year)
    if to_year is not None:
        where.append("p.year <= ?")
        params.append(to_year)
    if min_citations is not None:
        where.append("p.cited_by_count >= ?")
        params.append(min_citations)
    params.append(limit)

    sql = f"""
        SELECT p.id, p.title, p.year, p.venue, p.cited_by_count,
               -bm25(papers_fts, {TITLE_WEIGHT}, {ABSTRACT_WEIGHT}) AS score,
               snippet(papers_fts, 1, '{HL_OPEN}', '{HL_CLOSE}', ' … ', 24) AS snippet
        FROM papers_fts
        JOIN papers p ON p.rowid = papers_fts.rowid
        WHERE {' AND '.join(where)}
        ORDER BY score DESC
        LIMIT ?
    """
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        raise ValueError(f"Invalid search query {match!r}: {e}") from e
    return [
        SearchHit(
            id=r["id"], title=r["title"], year=r["year"], venue=r["venue"],
            cited_by_count=r["cited_by_count"], score=round(r["score"], 4), snippet=r["snippet"] or "",
        )
        for r in rows
    ]
