"""Command-line interface. Run `papercuts --help` for the list of commands."""

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


def _load_semantic(database: Database, model: str | None = None, need_model: bool = True):
    """Returns (embedder or None, index). Exits with a helpful message if nothing is embedded yet."""
    from papercuts.embeddings import SentenceTransformerEmbedder, VectorIndex

    name = model or Settings.from_env().embed_model
    index = VectorIndex.load(database, name)
    if not len(index):
        console.print(f"[yellow]No embeddings for model {name!r} yet.[/yellow] Run [bold]papercuts embed[/bold] first.")
        raise typer.Exit(1)
    if not need_model:
        return None, index
    with console.status(f"Loading embedding model {name} …"):
        try:
            embedder = SentenceTransformerEmbedder(name)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    return embedder, index


def _print_hits(hits, show_ranks: bool = False) -> None:
    if not hits:
        console.print("No matches.")
        return
    table = Table(show_lines=True)
    cols = [("#", {"justify": "right"}), ("Score", {"justify": "right"}), ("Year", {}),
            ("Cites", {"justify": "right"}), ("Title / snippet", {"ratio": 1}), ("ID", {"style": "dim"})]
    for col, kw in cols:
        table.add_column(col, **kw)
    for i, h in enumerate(hits, 1):
        snippet = escape(h.snippet).replace(HL_OPEN, "[bold yellow]").replace(HL_CLOSE, "[/bold yellow]")
        why = ""
        if show_ranks and h.ranks:
            why = "\n[cyan]" + " · ".join(f"{m} #{r}" for m, r in sorted(h.ranks.items())) + "[/cyan]"
        table.add_row(str(i), f"{h.score:.3f}", str(h.year or ""), str(h.cited_by_count),
                      f"[bold]{escape(h.title)}[/bold]\n[dim]{snippet}[/dim]{why}", h.id)
    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(...),
    limit: int = typer.Option(10, "--limit", "-k"),
    method: str = typer.Option("auto", "--method", "-m",
                               help="auto | bm25 | dense | hybrid (auto = hybrid if embeddings exist, else bm25)"),
    mode: str = typer.Option("any", "--mode", help="BM25 only: any (OR) or all (AND)"),
    raw: bool = typer.Option(False, "--raw", help="BM25 only: pass query straight to FTS5 syntax"),
    from_year: Optional[int] = typer.Option(None, "--from-year"),
    to_year: Optional[int] = typer.Option(None, "--to-year"),
    min_citations: Optional[int] = typer.Option(None, "--min-citations"),
    as_json: bool = typer.Option(False, "--json"),
    db: Optional[Path] = DbOpt,
):
    """Search the local corpus: keyword (BM25), meaning (dense), or both (hybrid)."""
    from papercuts.search import dense_search, hybrid_search

    filters = dict(from_year=from_year, to_year=to_year, min_citations=min_citations)
    with _open_db(db) as database:
        if method == "auto":
            has_vectors = database.conn.execute(
                "SELECT 1 FROM embeddings WHERE model = ? LIMIT 1", (Settings.from_env().embed_model,)
            ).fetchone()
            method = "hybrid" if has_vectors else "bm25"
        if method not in {"bm25", "dense", "hybrid"}:
            console.print("[red]--method must be auto, bm25, dense or hybrid[/red]")
            raise typer.Exit(1)
        try:
            if method == "bm25":
                hits = bm25_search(database.conn, query, limit=limit, mode=mode, raw=raw, **filters)
            else:
                embedder, index = _load_semantic(database)
                fn = dense_search if method == "dense" else hybrid_search
                hits = fn(database.conn, index, embedder, query, limit=limit, **filters)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    if as_json:
        print(json.dumps([h.__dict__ for h in hits], indent=2, ensure_ascii=False))
        return
    if not as_json:
        console.print(f"[dim]method: {method}[/dim]")
    _print_hits(hits, show_ranks=method == "hybrid")


@app.command()
def similar(
    paper_id: str = typer.Argument(..., help="openalex:W…, W…, or a DOI"),
    limit: int = typer.Option(10, "--limit", "-k"),
    from_year: Optional[int] = typer.Option(None, "--from-year"),
    db: Optional[Path] = DbOpt,
):
    """Papers most similar in meaning to one paper (uses stored embeddings; no model load)."""
    from papercuts.search import similar_papers

    with _open_db(db) as database:
        pid = database.resolve_id(paper_id)
        if pid is None:
            console.print(f"[red]Not found:[/red] {paper_id}")
            raise typer.Exit(1)
        _, index = _load_semantic(database, need_model=False)
        try:
            hits = similar_papers(database.conn, index, pid, limit=limit, from_year=from_year)
        except KeyError as e:
            console.print(f"[yellow]{e.args[0]}[/yellow]")
            raise typer.Exit(1)
        title = database.conn.execute("SELECT title FROM papers WHERE id = ?", (pid,)).fetchone()[0]
    console.print(f"Similar to: [bold]{escape(title)}[/bold]")
    _print_hits(hits)


@app.command()
def embed(
    model: Optional[str] = typer.Option(None, "--model", help="bge-small | bge-base | specter | any HF model id"),
    force: bool = typer.Option(False, "--force", help="Re-embed everything"),
    batch_size: int = typer.Option(32, "--batch-size"),
    db: Optional[Path] = DbOpt,
):
    """Compute embeddings for papers that are new or whose text changed."""
    from papercuts.embeddings import SentenceTransformerEmbedder, embed_corpus

    name = model or Settings.from_env().embed_model
    with console.status(f"Loading embedding model {name} (first time downloads it) …"):
        try:
            embedder = SentenceTransformerEmbedder(name, batch_size=batch_size)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    with _open_db(db) as database, Progress(
        TextColumn("[bold]Embedding"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console, transient=True,
    ) as progress:
        task = progress.add_task("embed", total=None)
        s = embed_corpus(database, embedder, batch_size=batch_size, force=force,
                         on_progress=lambda d, t: progress.update(task, completed=d, total=t))
    console.print(f"[green]Done.[/green] model {s.model} ({embedder.dim}-d) · embedded {s.embedded} · "
                  f"unchanged {s.skipped_unchanged} · total {s.total}")


@app.command()
def backfill(
    retry_missing: bool = typer.Option(False, "--retry", help="Also retry papers Semantic Scholar didn't have last time"),
    db: Optional[Path] = DbOpt,
):
    """Fill missing abstracts from Semantic Scholar (matched by DOI)."""
    from papercuts.enrich import backfill_abstracts
    from papercuts.sources.semantic_scholar import SemanticScholarClient, SemanticScholarError

    settings = Settings.from_env()

    async def run():
        with Database(db or settings.db_path) as database:
            async with SemanticScholarClient(settings.s2_api_key, timeout=max(settings.request_timeout, 60)) as client:
                return await backfill_abstracts(database, client)

    try:
        with console.status("Asking Semantic Scholar for missing abstracts …"):
            s = asyncio.run(run())
    except SemanticScholarError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    if s.candidates == 0:
        console.print("Nothing to backfill: every paper with a DOI has an abstract (or was already tried).")
        return
    console.print(f"[green]Filled {s.filled}[/green] of {s.candidates} missing abstracts · "
                  f"S2 has the paper but no abstract: {s.no_abstract_there} · not in S2: {s.not_found} · "
                  f"API calls {s.requests}")
    if s.filled:
        console.print("Run [bold]papercuts embed[/bold] to update embeddings for these papers.")


@app.command()
def snowball(
    min_citing: int = typer.Option(3, "--min-citing", help="Only fetch works cited by at least this many corpus papers"),
    limit: int = typer.Option(50, "--limit", "-n"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List candidates without fetching"),
    db: Optional[Path] = DbOpt,
):
    """Backward snowballing: add the papers your corpus cites most (free OpenAlex lookups)."""
    from papercuts.enrich import snowball_backward, snowball_candidates

    settings = Settings.from_env()
    if dry_run:
        with _open_db(db) as database:
            cands = snowball_candidates(database, min_citing, limit)
        table = Table(title=f"{len(cands)} works cited by ≥{min_citing} corpus papers but not in the corpus")
        table.add_column("Cited by", justify="right")
        table.add_column("OpenAlex id")
        for c in cands:
            table.add_row(str(c.cited_by_in_corpus), c.id)
        console.print(table)
        return

    async def run():
        with Database(db or settings.db_path) as database:
            async with OpenAlexClient(api_key=settings.openalex_api_key, mailto=settings.mailto,
                                      timeout=settings.request_timeout,
                                      max_concurrency=settings.max_concurrency) as client:
                with Progress(TextColumn("[bold]Snowballing"), BarColumn(), MofNCompleteColumn(),
                              TimeElapsedColumn(), console=console, transient=True) as progress:
                    task = progress.add_task("snowball", total=None)
                    return await snowball_backward(
                        database, client, min_citing=min_citing, limit=limit,
                        on_progress=lambda d, t: progress.update(task, completed=d, total=t),
                    )

    try:
        s = asyncio.run(run())
    except OpenAlexError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    if s.candidates == 0:
        console.print(f"No works are cited by ≥{min_citing} corpus papers without already being in it.")
        return
    console.print(f"[green]Run {s.run_id} done.[/green] candidates {s.candidates} · added {s.added} · "
                  f"failed {len(s.failed)} · API calls {s.requests}")
    console.print("Then: [bold]papercuts backfill[/bold] and [bold]papercuts embed[/bold] to cover the new papers.")


@app.command()
def dedup(
    threshold: float = typer.Option(0.93, "--threshold", min=0.5, max=1.0, help="Title similarity needed (0–1)"),
    merge: bool = typer.Option(False, "--merge", help="Merge the pairs found (after showing them)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation"),
    db: Optional[Path] = DbOpt,
):
    """Find (and optionally merge) the same paper stored twice, e.g. preprint + journal version."""
    from papercuts.enrich import find_duplicates

    with _open_db(db) as database:
        pairs = find_duplicates(database, threshold=threshold)
        if not pairs:
            console.print("No likely duplicates found.")
            return
        table = Table(title=f"{len(pairs)} likely duplicate pairs", show_lines=True)
        for col in ["Sim", "Reason", "Keep", "Drop"]:
            table.add_column(col)
        fmt = lambda p: f"{escape(p['title'])}\n[dim]{p['id']} · {p['type']} · {p['year']} · doi {p['doi'] or '—'}[/dim]"  # noqa: E731
        for d in pairs:
            table.add_row(f"{d.similarity:.2f}", d.reason, fmt(d.keep), fmt(d.drop))
        console.print(table)
        if not merge:
            console.print("Review them, then run [bold]papercuts dedup --merge[/bold] to merge.")
            return
        if not yes and not typer.confirm(f"Merge all {len(pairs)} pairs?"):
            raise typer.Exit(0)
        for d in pairs:
            database.merge_papers(d.keep["id"], d.drop["id"], reason=f"title match {d.similarity:.2f}: {d.reason}")
    console.print(f"[green]Merged {len(pairs)} pairs.[/green] Embeddings for kept papers will refresh on the next `papercuts embed`.")


def _eval_methods(database: Database):
    """Search functions for evaluation, sharing one loaded model."""
    from papercuts.search import dense_search, hybrid_search

    embedder, index = _load_semantic(database)
    conn = database.conn
    return {
        "bm25": lambda q, k: [h.id for h in bm25_search(conn, q, limit=k)],
        "dense": lambda q, k: [h.id for h in dense_search(conn, index, embedder, q, limit=k)],
        "hybrid": lambda q, k: [h.id for h in hybrid_search(conn, index, embedder, q, limit=k)],
    }


QrelsOpt = typer.Option(Path("eval/qrels.jsonl"), "--qrels", help="Relevance judgments file")


@app.command()
def judge(
    query: str = typer.Argument(..., help="A realistic search you'd run on this corpus"),
    depth: int = typer.Option(10, "--depth", help="Top-N from each method goes into the pool"),
    qrels: Path = QrelsOpt,
    db: Optional[Path] = DbOpt,
):
    """Rate pooled search results for one query (method hidden). Builds the evaluation set."""
    from papercuts.evaluation import load_qrels, pool, save_qrels

    all_qrels = load_qrels(qrels)
    judged = all_qrels.setdefault(query, {})
    with _open_db(db) as database:
        methods = _eval_methods(database)
        ids = [pid for pid in pool(methods, query, depth) if pid not in judged]
        if not ids:
            console.print(f"All pooled results for this query are already judged ({len(judged)}).")
            return
        console.print(f"[bold]{len(ids)}[/bold] papers to rate for: [italic]{escape(query)}[/italic]\n"
                      "  [bold]2[/bold] highly relevant · [bold]1[/bold] partly · [bold]0[/bold] not relevant · "
                      "[bold]s[/bold] skip · [bold]q[/bold] save & quit\n")
        for n, pid in enumerate(ids, 1):
            p = database.get_paper(pid)
            if p is None:
                continue
            abstract = (p["abstract"] or "(no abstract)")[:600]
            console.rule(f"{n}/{len(ids)}")
            console.print(f"[bold]{escape(p['title'])}[/bold] [dim]({p['year']})[/dim]\n{escape(abstract)}")
            while True:
                ans = typer.prompt("Relevance", default="s").strip().lower()
                if ans in {"0", "1", "2", "s", "q"}:
                    break
            if ans == "q":
                break
            if ans != "s":
                judged[pid] = int(ans)
                save_qrels(qrels, all_qrels)
    console.print(f"[green]Saved.[/green] {len(judged)} judgments for this query · {len(all_qrels)} queries in {qrels}")


@app.command(name="eval")
def eval_cmd(
    qrels: Path = QrelsOpt,
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Also write results as Markdown"),
    db: Optional[Path] = DbOpt,
):
    """Score bm25 vs dense vs hybrid on your judged queries (nDCG@10, Recall@20, MRR)."""
    from papercuts.evaluation import evaluate, load_qrels, results_markdown

    all_qrels = load_qrels(qrels)
    if not all_qrels:
        console.print(f"No judgments in {qrels}. Start with: [bold]papercuts judge \"your query\"[/bold]")
        raise typer.Exit(1)
    with _open_db(db) as database:
        scores = evaluate(_eval_methods(database), all_qrels)
    n_judged = sum(len(j) for j in all_qrels.values())
    table = Table(title=f"Search quality · {scores[0].n_queries} queries · {n_judged} judgments")
    for col in ["Method", "nDCG@10", "Recall@20", "MRR"]:
        table.add_column(col, justify="right" if col != "Method" else "left")
    best = max(scores, key=lambda s: s.ndcg10).method
    for s in scores:
        style = "bold green" if s.method == best else ""
        table.add_row(s.method, f"{s.ndcg10:.3f}", f"{s.recall20:.3f}", f"{s.mrr:.3f}", style=style)
    console.print(table)
    if scores[0].n_queries < 15:
        console.print(f"[yellow]Only {scores[0].n_queries} queries judged — aim for 15–20 before trusting the ranking.[/yellow]")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(results_markdown(scores, n_judged))
        console.print(f"Wrote {output}")


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
    if p.get("abstract_source") and p["abstract_source"] != p["source"]:
        console.print(f"[dim](abstract from {p['abstract_source']})[/dim]")


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
