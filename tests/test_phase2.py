"""Phase 2: migration, embeddings, dense/hybrid search, backfill, snowballing, dedup, evaluation."""

import json
import re
import sqlite3

import httpx
import numpy as np
import pytest

from papercuts.embeddings import VectorIndex, embed_corpus
from papercuts.enrich import (
    backfill_abstracts, find_duplicates, normalise_title, snowball_backward, snowball_candidates,
)
from papercuts.evaluation import (
    evaluate, load_qrels, mrr, ndcg_at_k, pool, recall_at_k, results_markdown, save_qrels,
)
from papercuts.models import Paper
from papercuts.search import bm25_search, dense_search, hybrid_search, rrf_fuse, similar_papers
from papercuts.sources.openalex import OpenAlexClient, parse_work
from papercuts.sources.semantic_scholar import SemanticScholarClient, SemanticScholarError
from papercuts.storage import Database
from papercuts.storage.db import MIGRATIONS
from tests.conftest import FakeOpenAlex, make_work

# ----------------------------------------------------------------------------- fake embedder

SYNONYMS = {"ocean": "marine", "sea": "marine", "algae": "phytoplankton", "algal": "phytoplankton"}


class FakeEmbedder:
    """Bag-of-words hashed into 128 dims, with a few synonyms folded together, so
    'ocean' and 'marine' land on the same dimension the way a real model would place them nearby."""

    name = "fake"
    dim = 128

    def __init__(self):
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z]+", text.lower()):
            tok = SYNONYMS.get(tok, tok)
            v[hash(tok) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        self.calls += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def add(db, title, abstract=None, i=0, **kw):
    p = Paper(id=f"openalex:W{900 + i}", source="openalex", title=title, abstract=abstract, **kw)
    db.upsert_paper(p)
    db.conn.commit()
    return p.id


# ----------------------------------------------------------------------------- migration


def test_v1_database_upgrades_in_place(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(MIGRATIONS[0])
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        "INSERT INTO papers (id, source, title, abstract, first_seen_at, updated_at) "
        "VALUES ('openalex:W1', 'openalex', 'Old paper', 'An abstract', 'x', 'x')"
    )
    conn.commit()
    conn.close()
    with Database(path) as db:
        assert db.schema_version == 2
        row = db.conn.execute("SELECT abstract_source FROM papers WHERE id = 'openalex:W1'").fetchone()
        assert row[0] == "openalex"
        assert db.stats()["embedded"] == 0
        assert bm25_search(db.conn, "abstract")  # FTS index survived


# ----------------------------------------------------------------------------- embeddings & search


@pytest.fixture
def corpus(db):
    add(db, "Microplastics alter marine phytoplankton carbon fixation", "Field study in coastal waters.", 1, year=2021)
    add(db, "Polystyrene beads reduce algae growth", "Laboratory exposure of diatoms.", 2, year=2019)
    add(db, "Soil plastisphere microbial communities", "Plastic debris in agricultural soils.", 3, year=2023)
    add(db, "Bird song classification with deep learning", "Bioacoustic monitoring.", 4, year=2020)
    return db


def test_embed_is_incremental(corpus):
    emb = FakeEmbedder()
    s1 = embed_corpus(corpus, emb)
    assert (s1.embedded, s1.total) == (4, 4)
    s2 = embed_corpus(corpus, emb)
    assert s2.embedded == 0 and s2.skipped_unchanged == 4
    corpus.conn.execute("UPDATE papers SET abstract = 'new text' WHERE id = 'openalex:W903'")
    corpus.conn.commit()
    assert embed_corpus(corpus, emb).embedded == 1  # only the changed paper
    assert embed_corpus(corpus, emb, force=True).embedded == 4
    idx = VectorIndex.load(corpus, "fake")
    assert len(idx) == 4 and idx.matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(idx.matrix, axis=1), 1.0, atol=1e-5)


def test_dense_finds_meaning_that_bm25_misses(corpus):
    emb = FakeEmbedder()
    embed_corpus(corpus, emb)
    idx = VectorIndex.load(corpus, "fake")
    assert not bm25_search(corpus.conn, "ocean")  # no paper contains the word
    hits = dense_search(corpus.conn, idx, emb, "ocean", limit=2)
    assert hits[0].id == "openalex:W901"  # "marine" paper
    assert hits[0].snippet.startswith("Field study")


def test_dense_respects_filters(corpus):
    emb = FakeEmbedder()
    embed_corpus(corpus, emb)
    idx = VectorIndex.load(corpus, "fake")
    hits = dense_search(corpus.conn, idx, emb, "plastic", limit=10, from_year=2022)
    assert [h.id for h in hits] == ["openalex:W903"]


def test_rrf_fuse():
    fused = rrf_fuse({"bm25": ["a", "b", "c"], "dense": ["b", "d"]}, k=60)
    assert fused[0][0] == "b"  # ranked by both methods
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0][2] == {"bm25": 2, "dense": 1}
    assert {pid for pid, _, _ in fused} == {"a", "b", "c", "d"}


def test_hybrid_combines_both(corpus):
    emb = FakeEmbedder()
    embed_corpus(corpus, emb)
    idx = VectorIndex.load(corpus, "fake")
    hits = hybrid_search(corpus.conn, idx, emb, "ocean microplastics", limit=4)
    assert hits[0].id == "openalex:W901"
    assert set(hits[0].ranks) == {"bm25", "dense"}


def test_similar_excludes_itself(corpus):
    embed_corpus(corpus, FakeEmbedder())
    idx = VectorIndex.load(corpus, "fake")
    hits = similar_papers(corpus.conn, idx, "openalex:W901", limit=3)
    assert "openalex:W901" not in [h.id for h in hits]
    assert len(hits) == 3
    with pytest.raises(KeyError):
        similar_papers(corpus.conn, idx, "openalex:W999", limit=3)


# ----------------------------------------------------------------------------- Semantic Scholar backfill


class FakeS2:
    def __init__(self, abstracts: dict[str, str | None], fail_first: int = 0):
        self.abstracts = abstracts  # doi -> abstract (None = known but no abstract); missing = unknown
        self.fail_first = fail_first
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_first:
            self.fail_first -= 1
            return httpx.Response(429, text="slow down")
        ids = json.loads(request.content)["ids"]
        out = []
        for i in ids:
            doi = i.removeprefix("DOI:")
            out.append(None if doi not in self.abstracts else {"paperId": "x", "abstract": self.abstracts[doi]})
        return httpx.Response(200, json=out)


async def _no_sleep(_):
    return None


async def test_backfill_fills_and_remembers_misses(db):
    add(db, "Has abstract", "Already here", 1, doi="10.1/a")
    add(db, "Missing one", None, 2, doi="10.1/b")
    add(db, "Missing two", None, 3, doi="10.1/c")
    add(db, "Missing three", None, 4, doi="10.1/d")
    add(db, "No DOI", None, 5)
    fake = FakeS2({"10.1/b": "Recovered abstract about zooplankton.", "10.1/c": None})
    async with SemanticScholarClient(transport=httpx.MockTransport(fake.handler), sleep=_no_sleep) as c:
        s = await backfill_abstracts(db, c)
    assert (s.candidates, s.filled, s.no_abstract_there, s.not_found) == (3, 1, 1, 1)
    p = db.get_paper("openalex:W902")
    assert p["abstract"].startswith("Recovered") and p["abstract_source"] == "semantic_scholar"
    assert bm25_search(db.conn, "zooplankton")[0].id == "openalex:W902"  # FTS updated via trigger
    assert db.stats()["abstracts_backfilled"] == 1
    async with SemanticScholarClient(transport=httpx.MockTransport(fake.handler), sleep=_no_sleep) as c:
        s2 = await backfill_abstracts(db, c)
    assert s2.candidates == 0  # misses are not re-queried
    assert len(fake.requests) == 1


async def test_backfill_retries_then_gives_clear_error(db):
    add(db, "Missing", None, 1, doi="10.1/b")
    fake = FakeS2({"10.1/b": "Text"}, fail_first=2)
    async with SemanticScholarClient(transport=httpx.MockTransport(fake.handler), sleep=_no_sleep) as c:
        assert (await backfill_abstracts(db, c)).filled == 1
    add(db, "Missing too", None, 2, doi="10.1/z")
    fake = FakeS2({}, fail_first=99)
    async with SemanticScholarClient(transport=httpx.MockTransport(fake.handler), sleep=_no_sleep,
                                     max_attempts=3) as c:
        with pytest.raises(SemanticScholarError, match="PAPERCUTS_S2_API_KEY"):
            await backfill_abstracts(db, c)


# ----------------------------------------------------------------------------- snowballing


def _seed_corpus_for_snowball(db):
    for i in range(10, 31):  # W1010..W1030; work i cites W(1000+i-3 .. i-1) and W999999
        db.upsert_paper(parse_work(make_work(i)))
    db.conn.commit()


def test_snowball_candidates_ranked(db):
    _seed_corpus_for_snowball(db)
    cands = snowball_candidates(db, min_citing=1, limit=10)
    assert [(c.id, c.cited_by_in_corpus) for c in cands] == [
        ("openalex:W999999", 21), ("openalex:W1009", 3), ("openalex:W1008", 2), ("openalex:W1007", 1),
    ]
    assert [c.id for c in snowball_candidates(db, min_citing=3)] == ["openalex:W999999", "openalex:W1009"]


async def test_snowball_adds_cited_works_and_remembers_404s(db):
    _seed_corpus_for_snowball(db)
    before = db.stats()["citation_edges_in_corpus"]
    fake = FakeOpenAlex(n_works=50)
    async with OpenAlexClient(transport=fake.transport(), sleep=_no_sleep, min_interval=0) as c:
        s = await snowball_backward(db, c, min_citing=1, limit=10)
    assert s.candidates == 4 and s.added == 3 and s.failed == ["openalex:W999999"]
    st = db.stats()
    assert st["papers"] == 24
    # 3+2+1 citations into the new papers now resolve, plus 3 among them (W1009→W1007/8, W1008→W1007)
    assert st["citation_edges_in_corpus"] == before + 9
    assert st["prisma"]["identified_from_citation_search"] == 3
    # the 404 is not retried; the new papers bring their own references (next snowball generation)
    assert "openalex:W999999" not in [c.id for c in snowball_candidates(db, min_citing=1)]
    assert db.list_runs()[0]["source"] == "openalex:snowball"


async def test_snowball_follows_merged_work_redirect(db):
    _seed_corpus_for_snowball(db)
    fake = FakeOpenAlex(n_works=50)
    fake.works[45]["id"] = "https://openalex.org/W5555"
    fake.aliases["W1009"] = 45  # OpenAlex merged W1009 into W5555
    async with OpenAlexClient(transport=fake.transport(), sleep=_no_sleep, min_interval=0) as c:
        await snowball_backward(db, c, min_citing=3, limit=10)
    assert db.get_paper("openalex:W5555")["n_cited_by_in_corpus"] == 3
    assert db.conn.execute("SELECT count(*) FROM citations WHERE cited_id = 'openalex:W1009'").fetchone()[0] == 0


# ----------------------------------------------------------------------------- dedup & merge


def test_normalise_title():
    assert normalise_title("Micro<i>plastics</i>: A Review — Part 1!") == "micro plastics a review part 1"
    assert normalise_title("Effets écologiques") == "effets ecologiques"


def test_find_and_merge_preprint_with_article(db):
    t = "Microplastics reduce carbon fixation in marine diatoms"
    add(db, t + ".", None, 1, type="preprint", year=2022, doi="10.1101/2022.01.01", cited_by_count=4)
    add(db, t, "Published abstract.", 2, type="article", year=2023, doi="10.1016/j.x.2023", cited_by_count=10)
    add(db, "Corrigendum to " + t, None, 3, type="article", year=2023)
    add(db, "Something completely different about soils", None, 4, year=2023)
    citing = Paper(id="openalex:W950", source="openalex", title="Citing paper", references=["openalex:W901"])
    db.upsert_paper(citing)
    run = db.start_run("q", {}, "openalex", 10)
    db.add_run_paper(run, "openalex:W901", 1)
    db.conn.commit()

    pairs = find_duplicates(db)
    assert len(pairs) == 1
    d = pairs[0]
    assert d.keep["id"] == "openalex:W902" and d.drop["id"] == "openalex:W901"
    assert d.reason == "preprint and published version"

    db.merge_papers(d.keep["id"], d.drop["id"], "test")
    assert db.get_paper("openalex:W901")["id"] == "openalex:W902"  # old id resolves via merge log
    kept = db.get_paper("openalex:W902")
    assert kept["cited_by_count"] == 14 and kept["n_cited_by_in_corpus"] == 1
    assert db.conn.execute("SELECT paper_id FROM run_papers").fetchone()[0] == "openalex:W902"
    assert db.stats()["prisma"]["duplicates_removed"] == 1
    assert find_duplicates(db) == []

    # Re-ingesting the dropped preprint must not resurrect it or overwrite the article.
    again = Paper(id="openalex:W901", source="openalex", title="PREPRINT TITLE", type="preprint", year=2022)
    assert db.upsert_paper(again) == ("openalex:W902", False)
    kept = db.get_paper("openalex:W902")
    assert kept["type"] == "article" and kept["title"] == t
    assert db.stats()["papers"] == 4


def test_same_doi_merge_is_logged(db):
    add(db, "A paper", "x", 1, doi="10.5/q")
    db.upsert_paper(Paper(id="s2:abc", source="semantic_scholar", title="A paper", doi="10.5/q"))
    assert db.stats()["duplicates_merged"] == 1
    assert db.resolve_id("s2:abc") == "openalex:W901"


# ----------------------------------------------------------------------------- evaluation


def test_metrics():
    j = {"a": 2, "b": 1, "c": 0, "d": 2}
    assert ndcg_at_k(["a", "d", "b"], j) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(["c", "b", "a"], j) < 0.6
    assert ndcg_at_k(["x"], {"x": 0}) == 0.0
    assert recall_at_k(["a", "zzz"], j, 20) == pytest.approx(1 / 3)
    assert mrr(["c", "x", "b"], j) == pytest.approx(1 / 3)


def test_qrels_roundtrip_and_pool(tmp_path):
    path = tmp_path / "eval" / "qrels.jsonl"
    save_qrels(path, {"q1": {"a": 2}, "q2": {"b": 0}})
    assert load_qrels(path) == {"q1": {"a": 2}, "q2": {"b": 0}}
    methods = {"m1": lambda q, k: ["a", "b", "c"][:k], "m2": lambda q, k: ["c", "d"][:k]}
    p = pool(methods, "q", depth=3)
    assert sorted(p) == ["a", "b", "c", "d"] and p == pool(methods, "q", depth=3)  # deterministic


def test_evaluate_ranks_methods():
    qrels = {"q1": {"a": 2, "b": 1}, "q2": {"c": 2}, "q3": {"z": 0}}  # q3 has no relevant docs: skipped
    methods = {"good": lambda q, k: {"q1": ["a", "b"], "q2": ["c"]}.get(q, []),
               "bad": lambda q, k: {"q1": ["x", "b"], "q2": ["y", "z", "c"]}.get(q, [])}
    scores = {s.method: s for s in evaluate(methods, qrels)}
    assert scores["good"].ndcg10 == pytest.approx(1.0) and scores["good"].n_queries == 2
    assert scores["bad"].ndcg10 < scores["good"].ndcg10
    md = results_markdown(list(scores.values()), 5)
    assert "| **good** | 1.000 |" in md


def test_dotenv_inline_comments(tmp_path, monkeypatch):
    from papercuts.config import _load_dotenv
    env = tmp_path / ".env"
    env.write_text('A_X=bge-small   # comment\nA_Y="has # hash"\nA_Z=\n')
    for k in ("A_X", "A_Y", "A_Z"):
        monkeypatch.delenv(k, raising=False)
    _load_dotenv(env)
    import os
    assert (os.environ["A_X"], os.environ["A_Y"], os.environ["A_Z"]) == ("bge-small", "has # hash", "")


# ----------------------------------------------------------------------------- real-model wrapper & CLI smoke


def test_sentence_transformer_wrapper_with_stub(monkeypatch):
    """The wrapper works with both the old and new sentence-transformers dimension API, and prefixes queries."""
    import sys
    import types

    seen = []

    class StubModel:
        def __init__(self, repo, device=None):
            self.repo = repo

        def get_embedding_dimension(self):  # sentence-transformers >= 6 name
            return 3

        def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
            seen.extend(texts)
            return np.ones((len(texts), 3)) / np.sqrt(3)

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=StubModel))
    from papercuts.embeddings import SentenceTransformerEmbedder

    e = SentenceTransformerEmbedder("bge-small")
    assert e.dim == 3 and e._model.repo == "BAAI/bge-small-en-v1.5"
    assert e.embed_documents(["doc"]).dtype == np.float32
    e.embed_query("ocean")
    assert seen == ["doc", "Represent this sentence for searching relevant passages: ocean"]


def test_cli_smoke(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import papercuts.embeddings as emb_mod
    from papercuts.cli import app

    dbp = tmp_path / "cli.db"
    monkeypatch.setenv("PAPERCUTS_EMBED_MODEL", "fake")
    monkeypatch.setattr(emb_mod, "SentenceTransformerEmbedder", lambda name, **kw: FakeEmbedder())
    with Database(dbp) as db:
        for i in range(10, 31):
            db.upsert_paper(parse_work(make_work(i)))
        db.conn.commit()

    run = CliRunner().invoke
    db_args = ["--db", str(dbp)]
    assert run(app, ["search", "carbon fixation", "-m", "bm25", *db_args]).exit_code == 0
    r = run(app, ["search", "carbon", *db_args])  # auto, nothing embedded yet -> bm25
    assert r.exit_code == 0 and "method: bm25" in r.output
    r = run(app, ["embed", *db_args])
    assert r.exit_code == 0 and "embedded 21" in r.output, r.output
    r = run(app, ["search", "ocean carbon", *db_args])
    assert r.exit_code == 0 and "method: hybrid" in r.output, r.output
    assert run(app, ["search", "x", "-m", "dense", *db_args]).exit_code == 0
    r = run(app, ["similar", "W1010", "-k", "3", *db_args])
    assert r.exit_code == 0 and "Similar to" in r.output, r.output
    r = run(app, ["snowball", "--dry-run", *db_args])
    assert r.exit_code == 0 and "W999999" in r.output
    assert run(app, ["dedup", *db_args]).exit_code == 0
    r = run(app, ["stats", *db_args])
    assert r.exit_code == 0 and "duplicates removed" in r.output

    qrels = tmp_path / "qrels.jsonl"
    r = run(app, ["judge", "carbon fixation", "--qrels", str(qrels), *db_args], input="2\n0\n1\nq\n")
    assert r.exit_code == 0, r.output
    assert sum(len(j) for j in load_qrels(qrels).values()) == 3
    out = tmp_path / "results.md"
    r = run(app, ["eval", "--qrels", str(qrels), "-o", str(out), *db_args])
    assert r.exit_code == 0 and "nDCG@10" in r.output and out.exists(), r.output
