"""Ingestion: run a source query and write results to the database, logged as a run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from papercuts.models import SearchFilters
from papercuts.sources.openalex import OpenAlexClient
from papercuts.storage.db import Database

COMMIT_EVERY = 100


@dataclass
class IngestSummary:
    run_id: int
    query: str
    total_available: int | None
    retrieved: int
    new: int
    updated: int
    requests: int


async def ingest_openalex(
    db: Database,
    client: OpenAlexClient,
    query: str,
    filters: SearchFilters | None = None,
    max_results: int = 200,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> IngestSummary:
    filters = filters or SearchFilters()
    run_id = db.start_run(query, filters.model_dump(), "openalex", max_results)
    retrieved = new = 0
    total: int | None = None
    try:
        rank = 0
        async for paper, total in client.search_works(query, filters, max_results):
            rank += 1
            stored_id, is_new = db.upsert_paper(paper)
            db.add_run_paper(run_id, stored_id, rank)
            retrieved += 1
            new += int(is_new)
            if retrieved % COMMIT_EVERY == 0:
                db.conn.commit()
            if on_progress:
                on_progress(retrieved, min(total, max_results) if total is not None else None)
        db.conn.commit()
    except BaseException as e:
        db.conn.commit()  # keep what we got; the run is marked failed and can be re-run
        db.finish_run(run_id, n_retrieved=retrieved, n_new=new, total_available=total,
                      status="failed", error=str(e)[:500])
        raise
    db.finish_run(run_id, n_retrieved=retrieved, n_new=new, total_available=total)
    return IngestSummary(
        run_id=run_id, query=query, total_available=total, retrieved=retrieved,
        new=new, updated=retrieved - new, requests=client.requests_made,
    )
