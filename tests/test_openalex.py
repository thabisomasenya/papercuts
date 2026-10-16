import pytest

from papercuts.models import SearchFilters, normalize_doi
from papercuts.sources.openalex import (
    OpenAlexBudgetError,
    OpenAlexClient,
    OpenAlexError,
    build_filter,
    parse_work,
    reconstruct_abstract,
)
from tests.conftest import FakeOpenAlex, make_work


def test_reconstruct_abstract_orders_words():
    assert reconstruct_abstract({"world": [1], "hello": [0], "again": [3], "hello,": [2]}) == "hello world hello, again"
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_build_filter():
    f = SearchFilters(from_year=2018, to_year=2024, types=["article", "review"], min_citations=10,
                      open_access_only=True, language="en")
    assert build_filter(f) == (
        "from_publication_date:2018-01-01,to_publication_date:2024-12-31,type:article|review,"
        "cited_by_count:>9,is_oa:true,language:en"
    )
    assert build_filter(SearchFilters()) is None


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1234/ABC") == "10.1234/abc"
    assert normalize_doi("doi:10.1/X") == "10.1/x"
    assert normalize_doi("not-a-doi") is None
    assert normalize_doi(None) is None


def test_parse_work():
    p = parse_work(make_work(5))
    assert p.id == "openalex:W1005"
    assert p.doi == "10.1234/test.5"
    assert p.abstract.startswith("We study")
    assert [a.id for a in p.authors] == ["openalex:A5", "openalex:A9999"]
    assert p.authors[0].institutions == ["University of Pretoria"]
    assert "openalex:W1004" in p.references and "openalex:W999999" in p.references
    assert "abstract_inverted_index" not in p.raw


async def test_paging_respects_max_and_per_page_limit(no_sleep):
    fake = FakeOpenAlex(n_works=250)
    async with OpenAlexClient(transport=fake.transport(), sleep=no_sleep, min_interval=0) as c:
        got = [p async for p, _ in c.search_works("microplastics", max_results=230)]
    assert len(got) == 230
    assert len({p.id for p in got}) == 230
    assert len(fake.requests) == 3  # 100 + 100 + 30
    assert all(int(r.url.params["per_page"]) <= 100 for r in fake.requests)
    assert fake.requests[-1].url.params["per_page"] == "30"


async def test_stops_when_results_run_out(no_sleep):
    fake = FakeOpenAlex(n_works=42)
    async with OpenAlexClient(transport=fake.transport(), sleep=no_sleep, min_interval=0) as c:
        got = [p async for p, total in c.search_works("x", max_results=500)]
    assert len(got) == 42


async def test_api_key_and_mailto_are_sent(no_sleep):
    fake = FakeOpenAlex(n_works=3)
    async with OpenAlexClient(api_key="k123", mailto="me@example.com", transport=fake.transport(),
                              sleep=no_sleep, min_interval=0) as c:
        [p async for p, _ in c.search_works("x", max_results=3)]
    params = fake.requests[0].url.params
    assert params["api_key"] == "k123" and params["mailto"] == "me@example.com"


async def test_retries_on_429_then_succeeds(no_sleep):
    fake = FakeOpenAlex(n_works=5, fail_first=[(429, {"Retry-After": "2"}), (503, {})])
    async with OpenAlexClient(transport=fake.transport(), sleep=no_sleep, min_interval=0) as c:
        got = [p async for p, _ in c.search_works("x", max_results=5)]
    assert len(got) == 5
    assert len(fake.requests) == 3


async def test_persistent_429_raises_budget_error(no_sleep):
    fake = FakeOpenAlex(n_works=5, fail_first=[(429, {})] * 10)
    async with OpenAlexClient(transport=fake.transport(), sleep=no_sleep, min_interval=0, max_attempts=3) as c:
        with pytest.raises(OpenAlexBudgetError, match="PAPERCUTS_OPENALEX_API_KEY"):
            [p async for p, _ in c.search_works("x", max_results=5)]
    assert len(fake.requests) == 3


async def test_client_error_is_not_retried(no_sleep):
    fake = FakeOpenAlex(n_works=5, fail_first=[(400, {})])
    async with OpenAlexClient(transport=fake.transport(), sleep=no_sleep, min_interval=0) as c:
        with pytest.raises(OpenAlexError, match="400"):
            [p async for p, _ in c.search_works("x", max_results=5)]
    assert len(fake.requests) == 1


async def test_get_work(no_sleep):
    fake = FakeOpenAlex(n_works=5)
    async with OpenAlexClient(transport=fake.transport(), sleep=no_sleep, min_interval=0) as c:
        p = await c.get_work("openalex:W1003")
    assert p.id == "openalex:W1003"


@pytest.mark.live
async def test_live_openalex():
    from papercuts.config import Settings
    s = Settings.from_env()
    async with OpenAlexClient(api_key=s.openalex_api_key, mailto=s.mailto) as c:
        got = [p async for p, _ in c.search_works("microplastics phytoplankton", max_results=5)]
    assert got and got[0].id.startswith("openalex:W")
