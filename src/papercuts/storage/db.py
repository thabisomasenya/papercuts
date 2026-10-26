"""SQLite storage: papers, authors, citations, run log, screening, and an FTS5 index.

One file on disk, WAL mode, schema versioned with PRAGMA user_version so later
phases can add tables through MIGRATIONS without breaking existing databases.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from papercuts.models import Paper

MIGRATIONS: list[str] = [
    # v1 — Phase 1 core
    """
    CREATE TABLE papers (
        id              TEXT PRIMARY KEY,
        source          TEXT NOT NULL,
        doi             TEXT,
        title           TEXT NOT NULL,
        abstract        TEXT,
        year            INTEGER,
        publication_date TEXT,
        venue           TEXT,
        type            TEXT,
        language        TEXT,
        cited_by_count  INTEGER NOT NULL DEFAULT 0,
        is_oa           INTEGER NOT NULL DEFAULT 0,
        oa_url          TEXT,
        topics          TEXT NOT NULL DEFAULT '[]',
        raw             TEXT NOT NULL DEFAULT '{}',
        first_seen_at   TEXT NOT NULL,
        updated_at      TEXT NOT NULL
    );
    CREATE UNIQUE INDEX idx_papers_doi ON papers(doi) WHERE doi IS NOT NULL;
    CREATE INDEX idx_papers_year ON papers(year);

    CREATE TABLE authors (
        id      TEXT PRIMARY KEY,
        name    TEXT NOT NULL,
        orcid   TEXT
    );

    CREATE TABLE paper_authors (
        paper_id     TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        author_id    TEXT NOT NULL REFERENCES authors(id),
        position     INTEGER NOT NULL,
        institutions TEXT NOT NULL DEFAULT '[]',
        PRIMARY KEY (paper_id, author_id)
    );
    CREATE INDEX idx_paper_authors_author ON paper_authors(author_id);

    -- cited_id may point outside the corpus; that is intentional (coverage stats).
    CREATE TABLE citations (
        citing_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        cited_id  TEXT NOT NULL,
        PRIMARY KEY (citing_id, cited_id)
    );
    CREATE INDEX idx_citations_cited ON citations(cited_id);

    CREATE TABLE runs (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        query           TEXT NOT NULL,
        filters         TEXT NOT NULL,
        source          TEXT NOT NULL,
        max_results     INTEGER NOT NULL,
        total_available INTEGER,
        n_retrieved     INTEGER NOT NULL DEFAULT 0,
        n_new           INTEGER NOT NULL DEFAULT 0,
        status          TEXT NOT NULL DEFAULT 'running',
        error           TEXT,
        started_at      TEXT NOT NULL,
        finished_at     TEXT
    );

    CREATE TABLE run_papers (
        run_id   INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        rank     INTEGER NOT NULL,
        PRIMARY KEY (run_id, paper_id)
    );

    -- Screening decisions (Phase 3). Created now so PRISMA counts work from day one.
    CREATE TABLE screening (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id    TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        stage       TEXT NOT NULL CHECK (stage IN ('title_abstract', 'full_text')),
        decision    TEXT NOT NULL CHECK (decision IN ('include', 'exclude', 'maybe')),
        reason      TEXT,
        decided_by  TEXT NOT NULL CHECK (decided_by IN ('human', 'model_suggestion')),
        decided_at  TEXT NOT NULL
    );
    CREATE INDEX idx_screening_paper ON screening(paper_id, stage);

    -- Full-text index (BM25). External content: mirrors papers via triggers.
    CREATE VIRTUAL TABLE papers_fts USING fts5(
        title, abstract,
        content='papers', content_rowid='rowid',
        tokenize='porter unicode61 remove_diacritics 2'
    );
    CREATE TRIGGER papers_ai AFTER INSERT ON papers BEGIN
        INSERT INTO papers_fts(rowid, title, abstract) VALUES (new.rowid, new.title, coalesce(new.abstract, ''));
    END;
    CREATE TRIGGER papers_ad AFTER DELETE ON papers BEGIN
        INSERT INTO papers_fts(papers_fts, rowid, title, abstract)
        VALUES ('delete', old.rowid, old.title, coalesce(old.abstract, ''));
    END;
    CREATE TRIGGER papers_au AFTER UPDATE OF title, abstract ON papers BEGIN
        INSERT INTO papers_fts(papers_fts, rowid, title, abstract)
        VALUES ('delete', old.rowid, old.title, coalesce(old.abstract, ''));
        INSERT INTO papers_fts(rowid, title, abstract) VALUES (new.rowid, new.title, coalesce(new.abstract, ''));
    END;
    """,
    # v2 — Phase 2: embeddings, enrichment bookkeeping, merge log
    """
    ALTER TABLE papers ADD COLUMN abstract_source TEXT;
    UPDATE papers SET abstract_source = source WHERE abstract IS NOT NULL AND abstract != '';

    CREATE TABLE embeddings (
        paper_id   TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        model      TEXT NOT NULL,
        dim        INTEGER NOT NULL,
        text_hash  TEXT NOT NULL,
        vector     BLOB NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (paper_id, model)
    );

    -- One row per (paper, enrichment source) so misses aren't re-queried every time.
    CREATE TABLE enrichment_attempts (
        paper_id     TEXT NOT NULL,
        source       TEXT NOT NULL,
        found        INTEGER NOT NULL,
        attempted_at TEXT NOT NULL,
        PRIMARY KEY (paper_id, source)
    );

    -- Every duplicate merge: same-DOI merges at ingest and title-based merges from `dedup`.
    -- Later ingests of a dropped id are redirected to the kept id.
    CREATE TABLE merges (
        dropped_id TEXT PRIMARY KEY,
        kept_id    TEXT NOT NULL,
        reason     TEXT NOT NULL,
        merged_at  TEXT NOT NULL
    );
    CREATE INDEX idx_merges_kept ON merges(kept_id);
    """,
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._migrate()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- schema ---------------------------------------------------------------------

    @property
    def schema_version(self) -> int:
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        current = self.schema_version
        for version, sql in enumerate(MIGRATIONS[current:], start=current + 1):
            with self.conn:
                self.conn.executescript(sql)
                self.conn.execute(f"PRAGMA user_version = {version}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.conn:
            yield self.conn

    # -- papers ---------------------------------------------------------------------

    def upsert_paper(self, paper: Paper) -> tuple[str, bool]:
        """Insert or update a paper and its authors/citations.

        Returns (stored_id, is_new). If another row already holds the same DOI
        (e.g. the same work from a different source), that row is updated and
        keeps its id, so the two records merge instead of duplicating.
        """
        c = self.conn
        now = _now()
        target_id = paper.id
        redirect = c.execute("SELECT kept_id FROM merges WHERE dropped_id = ?", (paper.id,)).fetchone()
        if redirect is not None:
            target_id = redirect["kept_id"]
        existing = c.execute("SELECT id FROM papers WHERE id = ?", (target_id,)).fetchone()
        if existing is None and paper.doi:
            existing = c.execute("SELECT id FROM papers WHERE doi = ?", (paper.doi,)).fetchone()
            if existing is not None:
                c.execute(
                    "INSERT OR IGNORE INTO merges (dropped_id, kept_id, reason, merged_at) VALUES (?, ?, ?, ?)",
                    (paper.id, existing["id"], "same DOI", now),
                )
        paper_id = existing["id"] if existing else target_id
        is_new = existing is None

        doi = paper.doi
        if doi:  # never let a DOI collide with a *different* stored paper
            holder = c.execute("SELECT id FROM papers WHERE doi = ?", (doi,)).fetchone()
            if holder is not None and holder["id"] != paper_id:
                doi = None

        values = dict(
            id=paper_id, source=paper.source, doi=doi, title=paper.title, abstract=paper.abstract,
            year=paper.year, publication_date=paper.publication_date, venue=paper.venue, type=paper.type,
            language=paper.language, cited_by_count=paper.cited_by_count, is_oa=int(paper.is_oa),
            oa_url=paper.oa_url, topics=json.dumps([t.model_dump() for t in paper.topics]),
            raw=json.dumps(paper.raw, default=str), now=now,
            abstract_source=paper.source if paper.abstract else None,
        )
        if existing is not None and paper_id != paper.id:
            # A different record of a paper we already hold (same DOI, or merged away earlier):
            # only fill gaps, never overwrite the canonical record's title/type/authors.
            c.execute(
                """
                UPDATE papers SET
                    abstract_source = CASE WHEN (abstract IS NULL OR abstract = '') AND :abstract IS NOT NULL
                                           THEN :abstract_source ELSE abstract_source END,
                    abstract = CASE WHEN (abstract IS NULL OR abstract = '') THEN :abstract ELSE abstract END,
                    year = coalesce(year, :year), venue = coalesce(venue, :venue),
                    oa_url = coalesce(oa_url, :oa_url), is_oa = max(is_oa, :is_oa), updated_at = :now
                WHERE id = :id
                """,
                values,
            )
        else:
            self._full_upsert(values, paper)
        self._add_references(paper_id, paper)
        if paper.id != paper_id:
            self._repoint_citations(paper.id, paper_id)
        return paper_id, is_new

    def _full_upsert(self, values: dict[str, Any], paper: Paper) -> None:
        c = self.conn
        paper_id = values["id"]
        c.execute(
            """
            INSERT INTO papers (id, source, doi, title, abstract, year, publication_date, venue, type, language,
                                cited_by_count, is_oa, oa_url, topics, raw, first_seen_at, updated_at,
                                abstract_source)
            VALUES (:id, :source, :doi, :title, :abstract, :year, :publication_date, :venue, :type, :language,
                    :cited_by_count, :is_oa, :oa_url, :topics, :raw, :now, :now, :abstract_source)
            ON CONFLICT(id) DO UPDATE SET
                doi = coalesce(excluded.doi, papers.doi),
                title = excluded.title,
                abstract_source = CASE WHEN excluded.abstract IS NOT NULL
                                       THEN excluded.abstract_source ELSE papers.abstract_source END,
                abstract = coalesce(excluded.abstract, papers.abstract),
                year = coalesce(excluded.year, papers.year),
                publication_date = coalesce(excluded.publication_date, papers.publication_date),
                venue = coalesce(excluded.venue, papers.venue),
                type = coalesce(excluded.type, papers.type),
                language = coalesce(excluded.language, papers.language),
                cited_by_count = max(excluded.cited_by_count, papers.cited_by_count),
                is_oa = max(excluded.is_oa, papers.is_oa),
                oa_url = coalesce(excluded.oa_url, papers.oa_url),
                topics = CASE WHEN excluded.topics != '[]' THEN excluded.topics ELSE papers.topics END,
                raw = excluded.raw,
                updated_at = excluded.updated_at
            """,
            values,
        )

        c.execute("DELETE FROM paper_authors WHERE paper_id = ?", (paper_id,))
        for a in paper.authors:
            c.execute(
                "INSERT INTO authors (id, name, orcid) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET name = excluded.name, orcid = coalesce(excluded.orcid, authors.orcid)",
                (a.id, a.name, a.orcid),
            )
            c.execute(
                "INSERT OR IGNORE INTO paper_authors (paper_id, author_id, position, institutions) VALUES (?, ?, ?, ?)",
                (paper_id, a.id, a.position, json.dumps(a.institutions)),
            )

    def _add_references(self, paper_id: str, paper: Paper) -> None:
        if not paper.references:
            return
        c = self.conn
        c.executemany(
            "INSERT OR IGNORE INTO citations (citing_id, cited_id) "
            "VALUES (?, coalesce((SELECT kept_id FROM merges WHERE dropped_id = ?), ?))",
            [(paper_id, ref, ref) for ref in paper.references if ref not in (paper_id, paper.id)],
        )
        c.execute("DELETE FROM citations WHERE citing_id = ? AND cited_id = ?", (paper_id, paper_id))

    def _repoint_citations(self, old_id: str, new_id: str) -> None:
        """Citations that pointed at old_id now point at new_id (no duplicates, no self-loops)."""
        c = self.conn
        c.execute("UPDATE OR IGNORE citations SET cited_id = ? WHERE cited_id = ?", (new_id, old_id))
        c.execute("DELETE FROM citations WHERE cited_id = ?", (old_id,))
        c.execute("DELETE FROM citations WHERE citing_id = cited_id")

    @staticmethod
    def _normalise_id(paper_id: str) -> str:
        if not paper_id.startswith(("openalex:", "s2:", "arxiv:")) and paper_id.upper().startswith("W"):
            return "openalex:" + paper_id.upper()
        return paper_id

    def resolve_id(self, paper_id: str) -> str | None:
        """Accepts openalex:W…, W…, a DOI, or an id that was merged away; returns the stored id."""
        pid = self._normalise_id(paper_id)
        row = self.conn.execute(
            "SELECT id FROM papers WHERE id = ? OR doi = lower(?) "
            "UNION ALL SELECT kept_id FROM merges WHERE dropped_id = ? LIMIT 1",
            (pid, pid, pid),
        ).fetchone()
        return row[0] if row else None

    def merge_papers(self, keep_id: str, drop_id: str, reason: str) -> None:
        """Fold drop_id into keep_id: fill gaps, move runs/citations/screening, log the merge."""
        if keep_id == drop_id:
            raise ValueError("cannot merge a paper into itself")
        c = self.conn
        keep = c.execute("SELECT * FROM papers WHERE id = ?", (keep_id,)).fetchone()
        drop = c.execute("SELECT * FROM papers WHERE id = ?", (drop_id,)).fetchone()
        if keep is None or drop is None:
            raise KeyError(f"unknown paper: {keep_id if keep is None else drop_id}")
        with c:
            c.execute("UPDATE papers SET doi = NULL WHERE id = ?", (drop_id,))  # free the unique DOI slot
            c.execute(
                """
                UPDATE papers SET
                    doi = coalesce(doi, :doi),
                    abstract_source = CASE WHEN (abstract IS NULL OR abstract = '') AND :abstract IS NOT NULL
                                           THEN :abstract_source ELSE abstract_source END,
                    abstract = CASE WHEN (abstract IS NULL OR abstract = '') THEN :abstract ELSE abstract END,
                    year = coalesce(year, :year),
                    venue = coalesce(venue, :venue),
                    oa_url = coalesce(oa_url, :oa_url),
                    is_oa = max(is_oa, :is_oa),
                    cited_by_count = cited_by_count + :cited_by_count,
                    updated_at = :now
                WHERE id = :keep_id
                """,
                dict(doi=drop["doi"], abstract=drop["abstract"] or None, abstract_source=drop["abstract_source"],
                     year=drop["year"], venue=drop["venue"], oa_url=drop["oa_url"], is_oa=drop["is_oa"],
                     cited_by_count=drop["cited_by_count"], now=_now(), keep_id=keep_id),
            )
            c.execute(
                "INSERT OR IGNORE INTO run_papers (run_id, paper_id, rank) "
                "SELECT run_id, ?, rank FROM run_papers WHERE paper_id = ?", (keep_id, drop_id),
            )
            c.execute(
                "INSERT OR IGNORE INTO citations (citing_id, cited_id) "
                "SELECT ?, cited_id FROM citations WHERE citing_id = ?", (keep_id, drop_id),
            )
            self._repoint_citations(drop_id, keep_id)
            c.execute("UPDATE screening SET paper_id = ? WHERE paper_id = ?", (keep_id, drop_id))
            c.execute(
                "INSERT OR IGNORE INTO paper_authors (paper_id, author_id, position, institutions) "
                "SELECT ?, author_id, position, institutions FROM paper_authors WHERE paper_id = ?",
                (keep_id, drop_id),
            )
            c.execute("UPDATE merges SET kept_id = ? WHERE kept_id = ?", (keep_id, drop_id))
            c.execute(
                "INSERT OR REPLACE INTO merges (dropped_id, kept_id, reason, merged_at) VALUES (?, ?, ?, ?)",
                (drop_id, keep_id, reason, _now()),
            )
            c.execute("DELETE FROM embeddings WHERE paper_id = ?", (keep_id,))  # text may have changed
            c.execute("DELETE FROM papers WHERE id = ?", (drop_id,))  # cascades run_papers, authors, embeddings

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        paper_id = self.resolve_id(paper_id) or paper_id
        row = self.conn.execute(
            "SELECT * FROM papers WHERE id = ? OR doi = lower(?)", (paper_id, paper_id)
        ).fetchone()
        if row is None:
            return None
        paper = dict(row)
        paper["topics"] = json.loads(paper["topics"])
        paper.pop("raw", None)
        paper["authors"] = [
            dict(r)
            for r in self.conn.execute(
                "SELECT a.id, a.name, a.orcid, pa.position FROM paper_authors pa "
                "JOIN authors a ON a.id = pa.author_id WHERE pa.paper_id = ? ORDER BY pa.position",
                (row["id"],),
            )
        ]
        paper["n_references"] = self.conn.execute(
            "SELECT count(*) FROM citations WHERE citing_id = ?", (row["id"],)
        ).fetchone()[0]
        paper["n_references_in_corpus"] = self.conn.execute(
            "SELECT count(*) FROM citations c JOIN papers p ON p.id = c.cited_id WHERE c.citing_id = ?",
            (row["id"],),
        ).fetchone()[0]
        paper["n_cited_by_in_corpus"] = self.conn.execute(
            "SELECT count(*) FROM citations WHERE cited_id = ?", (row["id"],)
        ).fetchone()[0]
        return paper

    # -- runs -----------------------------------------------------------------------

    def start_run(self, query: str, filters: dict[str, Any], source: str, max_results: int) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO runs (query, filters, source, max_results, started_at) VALUES (?, ?, ?, ?, ?)",
                (query, json.dumps(filters), source, max_results, _now()),
            )
        return int(cur.lastrowid)

    def add_run_paper(self, run_id: int, paper_id: str, rank: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO run_papers (run_id, paper_id, rank) VALUES (?, ?, ?)", (run_id, paper_id, rank)
        )

    def finish_run(
        self, run_id: int, *, n_retrieved: int, n_new: int, total_available: int | None,
        status: str = "done", error: str | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET n_retrieved = ?, n_new = ?, total_available = ?, status = ?, error = ?, "
                "finished_at = ? WHERE id = ?",
                (n_retrieved, n_new, total_available, status, error, _now(), run_id),
            )

    def list_runs(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM runs ORDER BY id DESC")]

    # -- stats ----------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        n_papers = q("SELECT count(*) FROM papers")
        return {
            "papers": n_papers,
            "with_abstract": q("SELECT count(*) FROM papers WHERE abstract IS NOT NULL AND abstract != ''"),
            "with_doi": q("SELECT count(*) FROM papers WHERE doi IS NOT NULL"),
            "open_access": q("SELECT count(*) FROM papers WHERE is_oa = 1"),
            "authors": q("SELECT count(*) FROM authors"),
            "citation_edges": q("SELECT count(*) FROM citations"),
            "citation_edges_in_corpus": q(
                "SELECT count(*) FROM citations c JOIN papers p ON p.id = c.cited_id"
            ),
            "abstracts_backfilled": q(
                "SELECT count(*) FROM papers WHERE abstract_source IS NOT NULL AND abstract_source != source"
            ),
            "embedded": q("SELECT count(DISTINCT paper_id) FROM embeddings"),
            "duplicates_merged": q("SELECT count(*) FROM merges"),
            "runs": q("SELECT count(*) FROM runs"),
            "year_min": q("SELECT min(year) FROM papers"),
            "year_max": q("SELECT max(year) FROM papers"),
            "prisma": self.prisma_counts(),
        }

    def prisma_counts(self) -> dict[str, int]:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        latest = (
            "SELECT s.* FROM screening s JOIN (SELECT paper_id, stage, max(id) AS mid FROM screening "
            "WHERE decided_by = 'human' GROUP BY paper_id, stage) l ON s.id = l.mid"
        )
        return {
            "identified_from_databases": q(
                "SELECT coalesce(sum(n_retrieved), 0) FROM runs WHERE status = 'done' AND source NOT LIKE '%snowball%'"
            ),
            "identified_from_citation_search": q(
                "SELECT coalesce(sum(n_retrieved), 0) FROM runs WHERE status = 'done' AND source LIKE '%snowball%'"
            ),
            "duplicates_removed": q("SELECT count(*) FROM merges"),
            "unique_records": q("SELECT count(DISTINCT paper_id) FROM run_papers"),
            "screened": q(f"SELECT count(*) FROM ({latest}) WHERE stage = 'title_abstract'"),
            "excluded_title_abstract": q(
                f"SELECT count(*) FROM ({latest}) WHERE stage = 'title_abstract' AND decision = 'exclude'"
            ),
            "included": q(f"SELECT count(*) FROM ({latest}) WHERE stage = 'full_text' AND decision = 'include'"),
        }
