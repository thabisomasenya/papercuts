"""Command-line interface: `papercuts ingest | search | show | stats | runs`."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from papercuts.config import Settings
from papercuts.ingest import ingest_openalex
from papercuts.models import SearchFilters
from papercuts.search import bm25_search
from papercuts.search.bm25 import HL_CLOSE, HL_OPEN
from papercuts.sources.openalex import OpenAlexClient, OpenAlexError
from papercuts.storage import Database

app = typer.Typer(add_completion=False, no_args_is_help=True, help="PaperCuts — evidence-mapping engine.")
console = Console()

DbOpt = typer.Option(None, "--db", help="SQLite file (default: $PAPERCUTS_DB or data/papercuts.db)")


def _open_db(db: Optional[Path]) -> Database:
    return Database(db or Settings.from_env().db_path)


@app.command()
def ingest(
    query: str = typer.Argument(..., help="Research question or keywords"),
    max_results: int = typer.Option(200, "--max", "-n", min=1, max=10_000),
    from_year: Optional[int] = typer.Option(None, "--from-year"),
    to_year: Optional[int] = typer.Option(None, "--to-year"),
    types: list[str] = typer.Option([], "--type", help="Work type, repeatable: article, review, preprint, ..."),
    min_citations: Optional[int] = typer.Option(None, "--min-citations"),
    oa_only: bool = typer.Option(False, "--oa-only", help="Only open-access works"),
    language: Optional[str] = typer.Option(None, "--lang", help="ISO 639-1 code, e.g. en"),
    db: Optional[Path] = DbOpt,
):
    """Fetch papers from OpenAlex and store them."""
    settings = Settings.from_env()
    filters = SearchFilters(
        from_year=from_year, to_year=to_year, types=types, min_citations=min_citations,
        open_access_only=oa_only, language=language,
    )

    async def run():
        with Database(db or settings.db_path) as database:
            async with OpenAlexClient(
                api_key=settings.openalex_api_key, mailto=settings.mailto,
                timeout=settings.request_timeout, max_concurrency=settings.max_concurrency,
            ) as client:
                with Progress(
                    TextColumn("[bold]Ingesting"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                    console=console, transient=True,
                ) as progress:
                    task = progress.add_task("ingest", total=max_results)

                    def on_progress(done: int, total: int | None) -> None:
                        progress.update(task, completed=done, total=total or max_results)

                    return await ingest_openalex(database, client, query, filters, max_results, on_progress)

    try:
        s = asyncio.run(run())
    except OpenAlexError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        f"[green]Run {s.run_id} done.[/green] OpenAlex matches: {s.total_available:,} · "
        f"retrieved {s.retrieved} · new {s.new} · updated {s.updated} · API calls {s.requests}"
        if s.total_available is not None
        else f"[yellow]Run {s.run_id}: no results for that query and filters.[/yellow]"
    )


@app.command()
def search(
    query: str = typer.Argument(...),
    limit: int = typer.Option(10, "--limit", "-k"),
    mode: str = typer.Option("any", "--mode", help="any (OR) or all (AND)"),
    raw: bool = typer.Option(False, "--raw", help="Pass query straight to FTS5 syntax"),
    from_year: Optional[int] = typer.Option(None, "--from-year"),
    to_year: Optional[int] = typer.Option(None, "--to-year"),
    min_citations: Optional[int] = typer.Option(None, "--min-citations"),
    as_json: bool = typer.Option(False, "--json"),
    db: Optional[Path] = DbOpt,
):
    """BM25 keyword search over the local corpus."""
    with _open_db(db) as database:
        try:
            hits = bm25_search(
                database.conn, query, limit=limit, mode=mode, raw=raw,
                from_year=from_year, to_year=to_year, min_citations=min_citations,
            )
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    if as_json:
        print(json.dumps([h.__dict__ for h in hits], indent=2, ensure_ascii=False))
        return
    if not hits:
        console.print("No matches.")
        return
    table = Table(show_lines=True)
    for col, kw in [("#", {"justify": "right"}), ("Score", {"justify": "right"}), ("Year", {}),
                    ("Cites", {"justify": "right"}), ("Title / snippet", {"ratio": 1}), ("ID", {"style": "dim"})]:
        table.add_column(col, **kw)
    for i, h in enumerate(hits, 1):
        snippet = escape(h.snippet).replace(HL_OPEN, "[bold yellow]").replace(HL_CLOSE, "[/bold yellow]")
        table.add_row(str(i), f"{h.score:.2f}", str(h.year or ""), str(h.cited_by_count),
                      f"[bold]{escape(h.title)}[/bold]\n[dim]{snippet}[/dim]", h.id)
    console.print(table)


@app.command()
def show(paper_id: str = typer.Argument(..., help="openalex:W…, W…, or a DOI"), db: Optional[Path] = DbOpt):
    """Show one paper in detail."""
    with _open_db(db) as database:
        p = database.get_paper(paper_id)
    if p is None:
        console.print(f"[red]Not found:[/red] {paper_id}")
        raise typer.Exit(1)
    authors = ", ".join(a["name"] for a in p["authors"][:8]) + (" et al." if len(p["authors"]) > 8 else "")
    console.print(f"[bold]{escape(p['title'])}[/bold]")
    console.print(escape(f"{authors}\n{p['venue'] or '—'} · {p['year'] or '—'} · {p['type'] or ''}"))
    console.print(f"DOI: {p['doi'] or '—'} · cited by {p['cited_by_count']} · "
                  f"{'open access' if p['is_oa'] else 'closed'}{' · ' + p['oa_url'] if p['oa_url'] else ''}")
    console.print(f"References: {p['n_references']} ({p['n_references_in_corpus']} in corpus) · "
                  f"cited by {p['n_cited_by_in_corpus']} papers in corpus")
    if p["topics"]:
        console.print("Topics: " + escape("; ".join(t["name"] for t in p["topics"])))
    console.print("\n" + (escape(p["abstract"]) if p["abstract"] else "[dim](no abstract available)[/dim]"))


@app.command()
def stats(db: Optional[Path] = DbOpt):
    """Corpus statistics and PRISMA counts."""
    with _open_db(db) as database:
        s = database.stats()
    prisma = s.pop("prisma")
    table = Table(title="Corpus", show_header=False)
    for k, v in s.items():
        text = "—" if v is None else str(v) if k.startswith("year") else f"{v:,}"
        table.add_row(k.replace("_", " "), text)
    console.print(table)
    pt = Table(title="PRISMA (from logged runs and screening)", show_header=False)
    for k, v in prisma.items():
        pt.add_row(k.replace("_", " "), f"{v:,}")
    console.print(pt)


@app.command()
def runs(db: Optional[Path] = DbOpt):
    """List ingestion runs."""
    with _open_db(db) as database:
        rows = database.list_runs()
    table = Table()
    for col in ["ID", "Status", "Query", "Filters", "Retrieved", "New", "Available", "Started"]:
        table.add_column(col)
    for r in rows:
        filters = {k: v for k, v in json.loads(r["filters"]).items() if v not in (None, [], False)}
        table.add_row(str(r["id"]), r["status"], r["query"], json.dumps(filters) if filters else "",
                      str(r["n_retrieved"]), str(r["n_new"]),
                      f"{r['total_available']:,}" if r["total_available"] is not None else "—", r["started_at"])
    console.print(table)


if __name__ == "__main__":
    app()
