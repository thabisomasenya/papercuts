# PaperCuts 

**Cut through the literature.** PaperCuts is an evidence-mapping engine for
academic research: give it a research question and it builds a deduplicated
corpus of papers, screens them, extracts structured evidence with quotes you
can check, maps the field as an interactive 2D/3D graph, and answers questions
about the corpus with citations.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-phase%201-orange)

See **[SPEC.md](SPEC.md)** for the full design.

## Roadmap

- [x] **Phase 1 — Corpus foundation:** OpenAlex ingestion, SQLite storage, BM25 search, run log
- [ ] **Phase 2 — Semantic layer:** local embeddings, hybrid search, cross-source dedup, search-quality eval
- [ ] **Phase 3 — Screening & extraction:** human-confirmed screening, structured extraction with evidence quotes, accuracy eval
- [ ] **Phase 4 — Map the field:** clustering and an interactive 2D/3D knowledge graph
- [ ] **Phase 5 — Ask the corpus:** RAG answers that cite their papers
- [ ] **Phase 6 — Synthesis:** claim-level agreement/contradiction, effect-size pooling, gap finder

## Setup (macOS / Linux)

```bash
git clone https://github.com/<you>/papercuts.git && cd papercuts
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # then add your OpenAlex key (free) and email
pytest                         # 18 offline tests, no network needed
```

Get a free OpenAlex API key at <https://openalex.org> (account settings). It works
without one, but the keyless daily budget is small — enough for about a
hundred keyword-search pages a day; a free key raises it roughly tenfold.

## Usage

```bash
papercuts ingest "microplastics phytoplankton carbon fixation" --max 200 --from-year 2015 --type article
papercuts search "carbon fixation" -k 10
papercuts search '"carbon fixation" diatoms' --mode all --from-year 2020
papercuts show W2741809807          # or a DOI: papercuts show 10.1038/...
papercuts stats                     # corpus + PRISMA counts
papercuts runs                      # every ingestion run with its query and filters
```

Running `ingest` again with the same or a related query is safe: papers are
upserted (citation counts refreshed, nothing duplicated) and each run is logged.

## Layout

```
src/papercuts/
  config.py            settings from env / .env
  models.py            Paper, Author, Topic, SearchFilters (source-independent)
  sources/openalex.py  async client: cursor paging, retries, Retry-After, budget errors
  storage/db.py        SQLite schema (versioned migrations), upserts, run log, PRISMA counts
  search/bm25.py       FTS5 BM25 search, title-weighted, injection-safe query builder
  ingest.py            source → database, logged as a run
  cli.py               typer CLI
tests/                 offline tests against a fake OpenAlex server; `pytest -m live` hits the real API
```

## Design decisions

- **SQLite + FTS5** instead of a server database: one file, BM25 built in, fast at
  this scale (tens of thousands of papers). Postgres + pgvector only if needed.
- **DOI is the merge key.** A paper arriving from another source with a known DOI
  updates the existing row instead of creating a duplicate.
- **Citations to papers outside the corpus are kept**, so we can measure how
  complete the corpus is and find highly-cited works we haven't ingested.
- **PRISMA counts are computed from logged runs and screening decisions**, never generated.

## Next up: Phase 2
Local embeddings (SPECTER2 / bge-small), dense + hybrid search (RRF), fuzzy
title dedup, Semantic Scholar backfill, and an nDCG@10 evaluation of BM25 vs
dense vs hybrid.
