"""Offline test fixtures: a fake OpenAlex /works endpoint built on httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

TOPICS = [
    ("microplastics reduce phytoplankton carbon fixation in marine mesocosms", "carbon fixation microplastics"),
    ("nanoplastic toxicity to diatoms and photosynthetic efficiency", "photosynthesis diatoms"),
    ("ocean acidification and coccolithophore calcification", "calcification acidification"),
    ("polystyrene particles alter algal growth rates", "algal growth polystyrene"),
    ("deep learning for bird song classification", "bioacoustics birds"),
]


def _inverted(text: str) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}
    for pos, word in enumerate(text.split()):
        idx.setdefault(word, []).append(pos)
    return idx


def make_work(i: int, *, doi: bool = True) -> dict:
    title, extra = TOPICS[i % len(TOPICS)]
    abstract = (
        f"We study {extra} in experiment {i}. Results show a significant effect on {extra.split()[0]} "
        f"compared with controls (n = {20 + i})."
    )
    return {
        "id": f"https://openalex.org/W{1000 + i}",
        "doi": f"https://doi.org/10.1234/Test.{i}" if doi else None,
        "display_name": f"{title.capitalize()} (study {i})",
        "publication_year": 2015 + (i % 10),
        "publication_date": f"{2015 + (i % 10)}-06-01",
        "primary_location": {"source": {"display_name": "Journal of Tests"}},
        "type": "article",
        "language": "en",
        "cited_by_count": i * 3,
        "open_access": {"is_oa": i % 2 == 0, "oa_url": f"https://example.org/{i}.pdf" if i % 2 == 0 else None},
        "authorships": [
            {"author_position": "first",
             "author": {"id": f"https://openalex.org/A{i}", "display_name": f"Author {i}", "orcid": None},
             "institutions": [{"display_name": "University of Pretoria"}]},
            {"author_position": "last",
             "author": {"id": "https://openalex.org/A9999", "display_name": "Senior Author", "orcid": None},
             "institutions": []},
        ],
        "topics": [{"id": "https://openalex.org/T1", "display_name": "Marine Ecology", "score": 0.9}],
        "referenced_works": [f"https://openalex.org/W{1000 + j}" for j in range(max(0, i - 3), i)]
                            + ["https://openalex.org/W999999"],
        "abstract_inverted_index": _inverted(abstract),
    }


class FakeOpenAlex:
    """Serves `n_works` results with cursor paging. Can inject failures."""

    def __init__(self, n_works: int = 250, fail_first: list[tuple[int, dict]] | None = None):
        self.works = [make_work(i) for i in range(n_works)]
        self.fail_first = list(fail_first or [])  # [(status, headers), ...] returned before real responses
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_first:
            status, headers = self.fail_first.pop(0)
            return httpx.Response(status, headers=headers, text="error")
        params = request.url.params
        if request.url.path.startswith("/works/"):
            key = request.url.path.rsplit("/", 1)[-1]
            for w in self.works:
                if w["id"].endswith(key):
                    return httpx.Response(200, json=w)
            return httpx.Response(404, text="not found")
        per_page = int(params.get("per_page", 25))
        if per_page > 100:
            return httpx.Response(400, json={"error": "per_page max is 100"})
        cursor = params.get("cursor")
        start = 0 if cursor in (None, "*") else int(cursor)
        page = self.works[start:start + per_page]
        nxt = start + per_page
        return httpx.Response(200, content=json.dumps({
            "meta": {"count": len(self.works), "per_page": per_page,
                     "next_cursor": str(nxt) if nxt < len(self.works) else None},
            "results": page,
        }))

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
def no_sleep():
    return _no_sleep


@pytest.fixture
def db(tmp_path):
    from papercuts.storage import Database
    d = Database(tmp_path / "test.db")
    yield d
    d.close()
