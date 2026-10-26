# PaperCuts — Spec

An evidence-mapping engine: give it a research question, it builds a screened,
deduplicated corpus of papers, extracts structured evidence from each one
(with quotes you can check), maps the field as a 2D/3D graph, and answers
questions about the corpus with citations.

**Design principles**

1. **Python engine, thin UI.** All data work lives in a Python package with a
   CLI and a FastAPI layer. The frontend only renders and calls the API.
2. **Free by default.** OpenAlex for metadata, local embedding models,
   SQLite on disk. Paid LLM calls are optional, batched and cached so each
   paper is processed once.
3. **Every claim is traceable.** Extracted fields carry an evidence quote and
   a source location. Nothing in the UI is an unsourced LLM summary.
4. **Measured, not assumed.** Extraction and contradiction detection are
   scored against a hand-labelled gold set, and the scores are in the README.
5. **Each phase ships a demo.** No phase depends on a later one to be useful.

---

## Architecture

```
            ┌──────────────────────────── papercuts (Python package) ───────────────────────────┐
query ──▶   sources/          dedup/         storage/          search/         analysis/
            openalex  ──┐                    SQLite             BM25 (FTS5)     extraction
            s2 (refs)  ─┼─▶  DOI + title ─▶  papers, authors,  embeddings ─▶   clustering
            arxiv     ──┘    fuzzy match     citations, runs,  hybrid (RRF)    claims/contradictions
                                             screening, cache                  effect-size pooling
            └──────────────────────────────────────┬─────────────────────────────────────────────┘
                                                   │ CLI (typer)  +  API (FastAPI)
                                                   ▼
                                     web/ (Next.js + react-force-graph)
                                     search · screening · matrix · graph · chat
```

| Layer | Choice | Why |
|---|---|---|
| Metadata | **OpenAlex** (primary), Semantic Scholar (citations/TLDR backfill), arXiv (preprints) | OpenAlex is free, broad, has abstracts, references and topics |
| Full text | Unpaywall → open-access PDF → **GROBID** (Phase 3b) | GROBID returns section-aware TEI; plain PDF text extraction doesn't |
| Storage | **SQLite** (WAL) + FTS5 | One file, zero ops, BM25 built in; swap to Postgres + pgvector only if needed |
| Embeddings | **SPECTER2** or `bge-small-en-v1.5` locally (sentence-transformers) | Free, good for scientific text; vectors stored as BLOBs, numpy cosine at this scale |
| Hybrid search | BM25 + dense, merged with **Reciprocal Rank Fusion** | Robust, no tuning needed |
| Clustering | UMAP → HDBSCAN, labels via c-TF-IDF (BERTopic-style) | Standard, explainable |
| LLM | Provider-agnostic interface; structured output with a Pydantic schema; responses cached in SQLite by `(paper_id, prompt_version, model)` | Swap models freely; never pay twice |
| Stats | `statsmodels` / own DerSimonian–Laird random-effects | Real meta-analysis where effect sizes exist |
| API | FastAPI | Typed, async, auto docs |
| UI | Next.js + Tailwind + Zustand + **react-force-graph** (2D/3D toggle) | Gets the 3D view without writing shaders |

---

## Data model (core tables)

- `papers` — id (OpenAlex W-id), doi, title, abstract, year, venue, type,
  language, cited_by_count, is_oa, oa_url, topics (JSON), raw (JSON), timestamps
- `authors`, `paper_authors` (with position)
- `citations` — (citing_id, cited_id); edges kept even when the cited paper
  isn't in the corpus, so coverage can be measured
- `runs` — each search: query, filters, source, timestamp, records identified
- `run_papers` — which run found which paper and at what rank
- `screening` — paper_id, stage (title_abstract / full_text), decision,
  reason, decided_by (human / model_suggestion), timestamp
- `papers_fts` — FTS5 index over title + abstract
- later: `embeddings`, `extractions`, `claims`, `llm_cache`

PRISMA counts are **queries over these tables**, not generated numbers:
identified = distinct papers across runs, duplicates removed = merges,
screened / excluded (with reasons) / included = `screening`.

---

## Phases

### Phase 1 — Corpus foundation  ✅
- Async OpenAlex client: cursor paging, polite pool (`mailto`), retry with
  backoff on 429/5xx, `Retry-After` respected, concurrency limit
- Abstract reconstruction from OpenAlex inverted index
- SQLite schema above (minus later tables), idempotent upserts
- Run logging for PRISMA "identified"
- BM25 search over title + abstract (title weighted higher), safe query parsing
- CLI: `ingest`, `search`, `show`, `stats`
- Tests run offline against recorded responses

**Demo:** `papercuts ingest "microplastics phytoplankton" --max 200` → `papercuts search "carbon fixation"`

### Phase 2 — Semantic layer  ✅
- Local embeddings for title + abstract (bge-small default); incremental via text hash; float32 BLOBs
- Dense search, hybrid search (RRF, k=60), similar-paper lookup
- Dedup: exact DOI at ingest, plus normalised-title fuzzy match (preprint ↔ published); merges logged and redirected
- Semantic Scholar backfill for missing abstracts (misses remembered)
- Backward citation snowballing (works cited by ≥N corpus papers), logged as its own run for PRISMA
- **Eval:** pooled, method-blind judging (`judge`) → nDCG@10 / Recall@20 / MRR for BM25 vs dense vs hybrid (`eval`)
- Later: forward snowballing (papers citing the corpus)

### Phase 3 — Screening and structured extraction
- Inclusion/exclusion criteria defined per project
- LLM *suggests* include/exclude with a reason; a human confirms in CLI/UI
- Extraction schema (Pydantic): objective, study_design (enum), population/system,
  sample_size, intervention/exposure, outcome, direction_of_effect,
  effect_size {value, type, ci_low, ci_high, n}, limitations — **each with an evidence quote**
- Abstract-first; full text via GROBID in 3b
- **Eval:** 30 hand-labelled papers → per-field accuracy; quote-in-source check catches hallucinations
- PRISMA flow diagram generated from the tables

### Phase 4 — Map the field
- UMAP (2D and 3D) + HDBSCAN clusters, auto-labelled
- Graph JSON: nodes (papers), edges = citation / similarity (top-k) / shared design
- FastAPI endpoints; Next.js app with search, paper drawer, filters
  (year, citations, design, cluster), 2D/3D toggle, click-to-focus

### Phase 5 — Ask the corpus
- RAG over abstracts/sections using hybrid search; answers must cite paper ids;
  uncited sentences are flagged
- Scoped questions: "within cluster 3", "only RCTs since 2020"

### Phase 6 — Synthesis
- **Claims:** extract outcome–direction claims per paper → embed → cluster similar claims →
  compare only within clusters (avoids O(n²) paper pairs) → agree / conflict / unclear, with quotes
- **Effect-size pooling:** where ≥3 papers report compatible effects, random-effects model,
  I², forest plot, funnel plot
- **Gap finder:** sparse regions in embedding space × low citation density × recent growth;
  suggestions are presented as leads with the papers that border them, not as conclusions
- Export: Markdown report, BibTeX, GraphML/JSON for Gephi

### Deferred (not in scope until 1–6 work)
Multi-user workspaces, shareable URLs with auth, annotations, Obsidian vault export, PDF viewer.

---

## Non-functional requirements
- Rate limits: shared async limiter per source; exponential backoff with jitter; resumable runs
- Reproducibility: every run stores query, filters, source and timestamp; LLM prompts are versioned
- Cost control: dry-run mode prints how many LLM calls a step would make before making them
- Config via environment / `.env` (`PAPERCUTS_DB`, `PAPERCUTS_MAILTO`, `PAPERCUTS_LLM_PROVIDER`, …)
- Tests: unit tests offline; one opt-in live test per source (`pytest -m live`)
