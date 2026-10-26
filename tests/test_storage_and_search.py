import pytest

from papercuts.ingest import ingest_openalex
from papercuts.models import Paper
from papercuts.search import bm25_search, build_fts_query
from papercuts.sources.openalex import OpenAlexBudgetError, OpenAlexClient, parse_work
from papercuts.storage import Database
from tests.conftest import FakeOpenAlex, make_work


def test_schema_is_versioned_and_reopens(tmp_path):
    path = tmp_path / "x.db"
    with Database(path) as d:
        assert d.schema_version == 2
    with Database(path) as d:  # re-opening must not re-run migrations
        assert d.schema_version == 2


def test_upsert_is_idempotent_and_updates(db):
    p = parse_work(make_work(1))
    assert db.upsert_paper(p) == ("openalex:W1001", True)
    p2 = p.model_copy(update={"cited_by_count": 500})
    assert db.upsert_paper(p2) == ("openalex:W1001", False)
    row = db.get_paper("W1001")
    assert row["cited_by_count"] == 500
    assert [a["name"] for a in row["authors"]] == ["Author 1", "Senior Author"]
    assert db.stats()["papers"] == 1


def test_same_doi_from_other_source_merges(db):
    db.upsert_paper(parse_work(make_work(2)))
    other = Paper(id="s2:abc", source="semantic_scholar", title="Same paper", doi="10.1234/TEST.2", abstract=None)
    stored_id, is_new = db.upsert_paper(other)
    assert (stored_id, is_new) == ("openalex:W1002", False)
    assert db.stats()["papers"] == 1
    assert db.get_paper("openalex:W1002")["abstract"]  # kept: new record had no abstract


def test_fts_follows_updates(db):
    p = parse_work(make_work(0))
    db.upsert_paper(p)
    assert bm25_search(db.conn, "carbon fixation")
    db.upsert_paper(p.model_copy(update={"title": "Entirely different heading", "abstract": "zebra"}))
    db.conn.commit()
    assert not bm25_search(db.conn, "carbon fixation")
    assert bm25_search(db.conn, "zebra")[0].id == p.id


def test_build_fts_query_is_safe():
    assert build_fts_query('effect of "carbon fixation" on algae NOT') == '"carbon fixation" OR "algae" OR "not"'
    assert build_fts_query("a AND b) OR (", mode="all") == '"b"'  # a/and/or are stopwords, parens dropped
    assert build_fts_query("the of") == '"the" OR "of"'  # all stopwords -> keep them
    assert build_fts_query("   ") == ""
    with pytest.raises(ValueError):
        build_fts_query("x", mode="maybe")


async def test_ingest_end_to_end(db, no_sleep):
    fake = FakeOpenAlex(n_works=150)
    async with OpenAlexClient(transport=fake.transport(), sleep=no_sleep, min_interval=0) as c:
        s = await ingest_openalex(db, c, "microplastics phytoplankton", max_results=120)
    assert (s.retrieved, s.new, s.total_available) == (120, 120, 150)

    async with OpenAlexClient(transport=fake.transport(), sleep=no_sleep, min_interval=0) as c:
        s2 = await ingest_openalex(db, c, "microplastics phytoplankton", max_results=150)
    assert (s2.retrieved, s2.new, s2.updated) == (150, 30, 120)

    st = db.stats()
    assert st["papers"] == 150
    assert st["prisma"]["identified_from_databases"] == 270  # every retrieved record, both runs
    assert st["prisma"]["unique_records"] == 150
    assert st["citation_edges_in_corpus"] > 0  # references to other ingested papers resolved

    hits = bm25_search(db.conn, "carbon fixation", limit=5)
    assert len(hits) == 5
    assert all("carbon fixation" in h.title.lower() for h in hits)  # title-weighted ranking
    assert "⟦" in hits[0].snippet

    hits_2020 = bm25_search(db.conn, "algal growth", from_year=2020, min_citations=100)
    assert hits_2020 and all(h.year >= 2020 and h.cited_by_count >= 100 for h in hits_2020)


async def test_failed_run_is_logged_and_keeps_partial_data(db, no_sleep):
    fake = FakeOpenAlex(n_works=250)
    async with OpenAlexClient(transport=fake.transport(), sleep=no_sleep, min_interval=0, max_attempts=2) as c:
        orig = fake.handler
        calls = {"n": 0}

        def flaky(request):
            calls["n"] += 1
            if calls["n"] >= 2:
                fake.fail_first = [(429, {})]
            return orig(request)

        fake.handler = flaky
        c._http._transport = fake.transport()
        with pytest.raises(OpenAlexBudgetError):
            await ingest_openalex(db, c, "x", max_results=250)
    run = db.list_runs()[0]
    assert run["status"] == "failed" and run["n_retrieved"] == 100
    assert db.stats()["papers"] == 100
