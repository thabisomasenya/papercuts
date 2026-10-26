"""Dense (embedding) search, hybrid BM25 + dense via Reciprocal Rank Fusion, and similar papers."""

from __future__ import annotations

import sqlite3

from papercuts.embeddings import Embedder, VectorIndex
from papercuts.search.bm25 import SearchHit, bm25_search

RRF_K = 60  # standard constant from Cormack et al. (2009); dampens the weight of top ranks
SNIPPET_CHARS = 220


def filter_ids(
    conn: sqlite3.Connection, from_year: int | None = None, to_year: int | None = None,
    min_citations: int | None = None,
) -> set[str] | None:
    """Ids passing the metadata filters, or None if no filter is set (= everything allowed)."""
    where, params = [], []
    if from_year is not None:
        where.append("year >= ?"); params.append(from_year)
    if to_year is not None:
        where.append("year <= ?"); params.append(to_year)
    if min_citations is not None:
        where.append("cited_by_count >= ?"); params.append(min_citations)
    if not where:
        return None
    return {r[0] for r in conn.execute(f"SELECT id FROM papers WHERE {' AND '.join(where)}", params)}


def _hydrate(
    conn: sqlite3.Connection, scored: list[tuple[str, float]],
    snippets: dict[str, str] | None = None, ranks: dict[str, dict[str, int]] | None = None,
) -> list[SearchHit]:
    if not scored:
        return []
    ids = [pid for pid, _ in scored]
    rows = {
        r["id"]: r for r in conn.execute(
            f"SELECT id, title, year, venue, cited_by_count, abstract FROM papers "
            f"WHERE id IN ({','.join('?' * len(ids))})", ids,
        )
    }
    hits = []
    for pid, score in scored:
        r = rows.get(pid)
        if r is None:  # vector for a paper that was since deleted/merged
            continue
        snippet = (snippets or {}).get(pid) or _lead(r["abstract"])
        hits.append(SearchHit(
            id=pid, title=r["title"], year=r["year"], venue=r["venue"], cited_by_count=r["cited_by_count"],
            score=round(score, 4), snippet=snippet, ranks=(ranks or {}).get(pid, {}),
        ))
    return hits


def _lead(abstract: str | None) -> str:
    if not abstract:
        return ""
    return abstract if len(abstract) <= SNIPPET_CHARS else abstract[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " …"


def dense_search(
    conn: sqlite3.Connection, index: VectorIndex, embedder: Embedder, query: str, *, limit: int = 20,
    from_year: int | None = None, to_year: int | None = None, min_citations: int | None = None,
) -> list[SearchHit]:
    allowed = filter_ids(conn, from_year, to_year, min_citations)
    scored = index.search(embedder.embed_query(query), k=limit, allowed=allowed)
    return _hydrate(conn, scored, ranks={pid: {"dense": i} for i, (pid, _) in enumerate(scored, 1)})


def rrf_fuse(rankings: dict[str, list[str]], k: int = RRF_K) -> list[tuple[str, float, dict[str, int]]]:
    """Reciprocal Rank Fusion: score(d) = Σ_methods 1 / (k + rank_m(d)). Rank-based, so no score calibration."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for method, ids in rankings.items():
        for rank, pid in enumerate(ids, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(pid, {})[method] = rank
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(pid, s, ranks[pid]) for pid, s in ordered]


def hybrid_search(
    conn: sqlite3.Connection, index: VectorIndex, embedder: Embedder, query: str, *, limit: int = 20,
    pool: int = 100, from_year: int | None = None, to_year: int | None = None, min_citations: int | None = None,
) -> list[SearchHit]:
    filters = dict(from_year=from_year, to_year=to_year, min_citations=min_citations)
    lexical = bm25_search(conn, query, limit=pool, **filters)
    allowed = filter_ids(conn, **filters)
    dense = index.search(embedder.embed_query(query), k=pool, allowed=allowed)
    fused = rrf_fuse({"bm25": [h.id for h in lexical], "dense": [pid for pid, _ in dense]})[:limit]
    snippets = {h.id: h.snippet for h in lexical}
    return _hydrate(conn, [(pid, s) for pid, s, _ in fused], snippets, {pid: r for pid, _, r in fused})


def similar_papers(
    conn: sqlite3.Connection, index: VectorIndex, paper_id: str, *, limit: int = 10,
    from_year: int | None = None, to_year: int | None = None, min_citations: int | None = None,
) -> list[SearchHit]:
    vec = index.vector(paper_id)
    if vec is None:
        raise KeyError(f"{paper_id} has no embedding for model {index.model!r}; run `papercuts embed` first")
    allowed = filter_ids(conn, from_year, to_year, min_citations)
    scored = index.search(vec, k=limit, allowed=allowed, exclude={paper_id})
    return _hydrate(conn, scored)
