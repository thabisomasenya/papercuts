"""Async OpenAlex client.

Notes on the API (developers.openalex.org):
- `per_page` max is 100; cursor paging (`cursor=*`) is needed beyond 10,000 results.
- Usage is metered in daily credits. Without a key the budget is small; a free key
  (`api_key` query param) raises it. Keyword search costs more than list/filter calls,
  and fetching a single work by id is free.
- Over-budget and over-rate requests both return 429, so after retries are exhausted
  we raise OpenAlexBudgetError with a hint instead of a bare HTTP error.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
)

from papercuts.models import Author, Paper, SearchFilters, Topic

BASE_URL = "https://api.openalex.org"
MAX_PER_PAGE = 100
SELECT_FIELDS = ",".join(
    [
        "id", "doi", "display_name", "publication_year", "publication_date",
        "primary_location", "type", "language", "cited_by_count", "open_access",
        "authorships", "topics", "referenced_works", "abstract_inverted_index",
    ]
)
RETRY_STATUSES = {429, 500, 502, 503, 504}


class OpenAlexError(RuntimeError):
    pass


class OpenAlexBudgetError(OpenAlexError):
    pass


class _RetryableHTTPError(Exception):
    def __init__(self, status: int, retry_after: float | None, body: str):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after
        self.body = body


# ----------------------------------------------------------------------------- helpers


def openalex_short_id(url_or_id: str) -> str:
    """'https://openalex.org/W123' -> 'W123'."""
    return url_or_id.rstrip("/").rsplit("/", 1)[-1]


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the text."""
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def build_filter(filters: SearchFilters) -> str | None:
    parts: list[str] = []
    if filters.from_year:
        parts.append(f"from_publication_date:{filters.from_year}-01-01")
    if filters.to_year:
        parts.append(f"to_publication_date:{filters.to_year}-12-31")
    if filters.types:
        parts.append("type:" + "|".join(filters.types))
    if filters.min_citations is not None and filters.min_citations > 0:
        parts.append(f"cited_by_count:>{filters.min_citations - 1}")
    if filters.open_access_only:
        parts.append("is_oa:true")
    if filters.language:
        parts.append(f"language:{filters.language}")
    return ",".join(parts) or None


def parse_work(work: dict[str, Any]) -> Paper:
    """Normalise one OpenAlex work into our Paper model."""
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    oa = work.get("open_access") or {}

    authors = []
    for pos, a in enumerate(work.get("authorships") or []):
        info = a.get("author") or {}
        if not info.get("id"):
            continue
        authors.append(
            Author(
                id="openalex:" + openalex_short_id(info["id"]),
                name=info.get("display_name") or "Unknown",
                orcid=info.get("orcid"),
                position=pos,
                institutions=[
                    i.get("display_name") for i in a.get("institutions") or [] if i.get("display_name")
                ],
            )
        )

    topics = [
        Topic(id="openalex:" + openalex_short_id(t["id"]), name=t.get("display_name", ""), score=t.get("score"))
        for t in work.get("topics") or []
        if t.get("id")
    ]

    return Paper(
        id="openalex:" + openalex_short_id(work["id"]),
        source="openalex",
        title=work.get("display_name") or work.get("title"),
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        doi=work.get("doi"),
        year=work.get("publication_year"),
        publication_date=work.get("publication_date"),
        venue=source.get("display_name"),
        type=work.get("type"),
        language=work.get("language"),
        cited_by_count=work.get("cited_by_count") or 0,
        is_oa=bool(oa.get("is_oa")),
        oa_url=oa.get("oa_url"),
        authors=authors,
        topics=topics,
        references=["openalex:" + openalex_short_id(r) for r in work.get("referenced_works") or []],
        raw={k: v for k, v in work.items() if k != "abstract_inverted_index"},
    )


def _wait(state: RetryCallState) -> float:
    """Honour Retry-After when given, else exponential backoff with jitter (cap 60s)."""
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, _RetryableHTTPError) and exc.retry_after is not None:
        return min(exc.retry_after, 120.0)
    return min(60.0, (2 ** (state.attempt_number - 1)) + random.uniform(0, 1))


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (_RetryableHTTPError, httpx.TransportError))


# ----------------------------------------------------------------------------- client


class OpenAlexClient:
    def __init__(
        self,
        api_key: str | None = None,
        mailto: str | None = None,
        *,
        timeout: float = 30.0,
        max_concurrency: int = 4,
        min_interval: float = 0.1,
        max_attempts: int = 6,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep=asyncio.sleep,
    ):
        self.api_key = api_key
        self.mailto = mailto
        self.max_attempts = max_attempts
        self._sem = asyncio.Semaphore(max_concurrency)
        self._min_interval = min_interval
        self._last_request = 0.0
        self._lock = asyncio.Lock()
        self._sleep = sleep
        self.requests_made = 0
        ua = "PaperCuts/0.1" + (f" (mailto:{mailto})" if mailto else "")
        self._http = httpx.AsyncClient(
            base_url=BASE_URL, timeout=timeout, transport=transport, headers={"User-Agent": ua}
        )

    async def __aenter__(self) -> "OpenAlexClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # -- low level ----------------------------------------------------------------

    async def _throttle(self) -> None:
        async with self._lock:
            delta = time.monotonic() - self._last_request
            if delta < self._min_interval:
                await self._sleep(self._min_interval - delta)
            self._last_request = time.monotonic()

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {k: v for k, v in params.items() if v is not None}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.mailto:
            params["mailto"] = self.mailto

        async def attempt() -> dict[str, Any]:
            async with self._sem:
                await self._throttle()
                self.requests_made += 1
                resp = await self._http.get(path, params=params)
            if resp.status_code in RETRY_STATUSES:
                ra = resp.headers.get("Retry-After")
                raise _RetryableHTTPError(
                    resp.status_code, float(ra) if ra and ra.replace(".", "", 1).isdigit() else None, resp.text[:300]
                )
            if resp.status_code >= 400:
                raise OpenAlexError(f"OpenAlex {resp.status_code} for {path}: {resp.text[:300]}")
            return resp.json()

        try:
            async for att in AsyncRetrying(
                retry=retry_if_exception(_is_retryable),
                wait=_wait,
                stop=stop_after_attempt(self.max_attempts),
                sleep=self._sleep,
                reraise=True,
            ):
                with att:
                    return await attempt()
        except _RetryableHTTPError as e:
            if e.status == 429:
                hint = "" if self.api_key else " Set PAPERCUTS_OPENALEX_API_KEY (free key) to raise the limit."
                raise OpenAlexBudgetError(
                    f"OpenAlex kept returning 429 after {self.max_attempts} attempts: "
                    f"rate limit or daily budget exhausted.{hint}"
                ) from e
            raise OpenAlexError(f"OpenAlex {e.status} after {self.max_attempts} attempts: {e.body}") from e
        except httpx.TransportError as e:
            raise OpenAlexError(f"Network error talking to OpenAlex: {e!r}") from e
        raise AssertionError("unreachable")  # pragma: no cover

    # -- public API -----------------------------------------------------------------

    async def count(self, query: str, filters: SearchFilters | None = None) -> int:
        data = await self._get(
            "/works",
            {"search": query, "filter": build_filter(filters or SearchFilters()), "per_page": 1, "select": "id"},
        )
        return int(data.get("meta", {}).get("count", 0))

    async def search_works(
        self,
        query: str,
        filters: SearchFilters | None = None,
        max_results: int = 200,
        sort: str | None = None,
    ) -> AsyncIterator[tuple[Paper, int]]:
        """Yield (paper, total_available) in relevance order, using cursor paging."""
        filters = filters or SearchFilters()
        cursor: str | None = "*"
        yielded = 0
        while cursor and yielded < max_results:
            data = await self._get(
                "/works",
                {
                    "search": query,
                    "filter": build_filter(filters),
                    "per_page": min(MAX_PER_PAGE, max_results - yielded),
                    "cursor": cursor,
                    "select": SELECT_FIELDS,
                    "sort": sort,
                },
            )
            total = int(data.get("meta", {}).get("count", 0))
            results = data.get("results") or []
            if not results:
                break
            for work in results:
                if yielded >= max_results:
                    break
                yield parse_work(work), total
                yielded += 1
            cursor = data.get("meta", {}).get("next_cursor")

    async def get_work(self, work_id: str) -> Paper:
        """Fetch one work by OpenAlex id or DOI (singleton lookups are free)."""
        key = work_id.split(":", 1)[1] if work_id.startswith("openalex:") else work_id
        if key.startswith("10."):
            key = "doi:" + key
        data = await self._get(f"/works/{key}", {"select": SELECT_FIELDS})
        return parse_work(data)
