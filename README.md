# PaperCuts 

**Cut through the literature.** PaperCuts is an evidence-mapping engine for
academic research: give it a research question and it builds a deduplicated
corpus of papers, screens them, extracts structured evidence with quotes you
can check, maps the field as an interactive 2D/3D graph, and answers questions
about the corpus with citations.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-phase%202-orange)

See **[SPEC.md](SPEC.md)** for the full design.

## Roadmap

- [x] **Phase 1 — Corpus foundation:** OpenAlex ingestion, SQLite storage, BM25 search, run log
- [x] **Phase 2 — Semantic layer:** local embeddings, hybrid search, abstract backfill, citation snowballing, dedup, search-quality eval
- [ ] **Phase 3 — Screening & extraction:** human-confirmed screening, structured extraction with evidence quotes, accuracy eval
- [ ] **Phase 4 — Map the field:** clustering and an interactive 2D/3D knowledge graph
- [ ] **Phase 5 — Ask the corpus:** RAG answers that cite their papers
- [ ] **Phase 6 — Synthesis:** claim-level agreement/contradiction, effect-size pooling, gap finder

## Setup (macOS / Linux)

```bash
git clone https://github.com/thabisomasenya/papercuts.git && cd papercuts
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,semantic]"   # drop ",semantic" if you only need keyword search
cp .env.example .env              # then add your OpenAlex key (free) and email
pytest                            # offline tests, no network needed
```

Get a free OpenAlex API key at <https://openalex.org> (account settings). It works
without one, but the keyless daily budget is small — enough for about a
hundred keyword-search pages a day; a free key raises it roughly tenfold.

## Usage

```bash
# 1. Build the corpus
papercuts ingest "microplastics phytoplankton carbon fixation" --max 200 --from-year 2015 --type article
papercuts snowball --dry-run        # works your corpus cites most but doesn't contain
papercuts snowball                  # fetch them (single-work lookups are free on OpenAlex)
papercuts backfill                  # fill missing abstracts from Semantic Scholar
papercuts dedup                     # preprint + journal versions of the same paper; add --merge
papercuts embed                     # local embeddings (first run downloads the model, ~130 MB)

# 2. Explore
papercuts search "ocean warming and plankton"          # hybrid by default once embedded
papercuts search "carbon fixation" -m bm25             # keyword only (also: -m dense)
papercuts search '"carbon fixation" diatoms' -m bm25 --mode all --from-year 2020
papercuts similar W2741809807 -k 10                    # papers closest in meaning to one paper
papercuts show W2741809807                             # or a DOI
papercuts stats                                        # corpus + PRISMA counts
papercuts runs                                         # every ingestion/snowball run

# 3. Measure search quality
papercuts judge "effects of microplastics on photosynthesis"   # rate pooled results 0/1/2, method hidden
papercuts eval -o eval/results.md                              # nDCG@10, Recall@20, MRR per method
```

Every command is safe to re-run: papers are upserted, embeddings are only
recomputed when a paper's text changes, and misses (no abstract on Semantic
Scholar, 404s while snowballing) are remembered so they aren't re-queried.

## How search works

| Method | What it matches | Good at |
|---|---|---|
| `bm25` | Exact words (stemmed), title weighted 3× | Precise terms, names, acronyms |
| `dense` | Meaning, via `bge-small-en-v1.5` embeddings | Synonyms and paraphrases ("ocean" ↔ "marine") |
| `hybrid` | Both, merged with Reciprocal Rank Fusion | Usually the best of both; shows each paper's rank per method |

Embedding models: `bge-small` (default, fast), `bge-base` (better, slower), `specter`
(trained on scientific citations). Set `PAPERCUTS_EMBED_MODEL` in `.env`.

## Search evaluation

`judge` pools the top 10 from every method, hides which method found each paper,
and asks you to rate relevance; judgments are saved to `eval/qrels.jsonl`.
`eval` then scores each method. Results (to be filled in once 15–20 queries are judged):

<!-- paste eval/results.md here -->

## Layout

```
src/papercuts/
  config.py                    settings from env / .env
  models.py                    Paper, Author, Topic, SearchFilters (source-independent)
  sources/openalex.py          async client: cursor paging, retries, Retry-After, budget errors
  sources/semantic_scholar.py  batch DOI lookup for abstract backfill
  storage/db.py                SQLite schema (versioned migrations), upserts, merges, run log, PRISMA counts
  search/bm25.py               FTS5 BM25 search, title-weighted, injection-safe query builder
  search/hybrid.py             dense search, RRF hybrid, similar papers
  embeddings.py                local embedding models, incremental embedding, vector index
  enrich.py                    abstract backfill, backward snowballing, near-duplicate detection
  evaluation.py                pooled judging, nDCG / Recall / MRR
  ingest.py                    source → database, logged as a run
  cli.py                       typer CLI
tests/                 offline tests against a fake OpenAlex server; `pytest -m live` hits the real API
```

## Design decisions

- **SQLite + FTS5** instead of a server database: one file, BM25 built in, fast at
  this scale (tens of thousands of papers). Postgres + pgvector only if needed.
- **DOI is the merge key.** A paper arriving from another source with a known DOI
  updates the existing row instead of creating a duplicate.
- **Citations to papers outside the corpus are kept**, so we can measure how
  complete the corpus is and find highly-cited works we haven't ingested.
- **PRISMA counts are computed from logged runs, merges and screening decisions**, never generated.
- **Merges are logged, not just deleted.** A merged-away id keeps resolving to the kept
  paper, and re-ingesting it later can't resurrect the duplicate.
- **Vectors live in SQLite** as float32 blobs with brute-force cosine search: exact,
  and fast up to roughly 100k papers. No vector database to run.

## Next up: Phase 3 — screening & extraction
Inclusion/exclusion criteria, LLM-suggested and human-confirmed screening, and
structured extraction (design, sample, outcome, effect size) where every field
carries an evidence quote, scored against a hand-labelled set.
