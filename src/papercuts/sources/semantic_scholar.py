"""Minimal async Semantic Scholar client: batch lookup of papers by DOI.

Used to backfill abstracts that OpenAlex doesn't have. Without an API key requests
go through a shared, heavily used pool and 429s are common, so we pace requests
(~1/s) and retry with backoff. A free key (x-api-key header) makes it reliable.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import httpx
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt

BASE_URL = "https://api.semanticscholar.org/graph/v1"
MAX_BATCH = 500


class SemanticScholarError(RuntimeError):
    pass


class _Retryable(Exception):
    def __init__(self, status: int, retry_after: float | None):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


def _wait(state: RetryCallState) -> float:
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, _Retryable) and exc.retry_after is not None:
        return min(exc.retry_after, 120.0)
    return min(60.0, 2 ** state.attempt_number + random.uniform(0, 1))


class SemanticScholarClient:
    def __init__(
        self, api_key: str | None = None, *, timeout: float = 60.0, min_interval: float = 1.1,
        max_attempts: int = 6, transport: httpx.AsyncBaseTransport | None = None, sleep=asyncio.sleep,
    ):
        headers = {"User-Agent": "PaperCuts/0.2"}
        if api_key:
            headers["x-api-key"] = api_key
        self.has_key = bool(api_key)
        self._http = httpx.AsyncClient(base_url=BASE_URL, timeout=timeout, transport=transport, headers=headers)
        self._min_interval = min_interval
        self._last = 0.0
        self._sleep = sleep
        self.max_attempts = max_attempts
        self.requests_made = 0

    async def __aenter__(self) -> "SemanticScholarClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._http.aclose()

    async def batch_by_doi(self, dois: list[str], fields: str = "title,abstract,externalIds") -> list[dict | None]:
        """One result per DOI, in order; None where Semantic Scholar doesn't know the paper."""
        if len(dois) > MAX_BATCH:
            raise ValueError(f"at most {MAX_BATCH} ids per batch")
        if not dois:
            return []

        async def attempt() -> list[dict[str, Any] | None]:
            delta = time.monotonic() - self._last
            if delta < self._min_interval:
                await self._sleep(self._min_interval - delta)
            self._last = time.monotonic()
            self.requests_made += 1
            resp = await self._http.post("/paper/batch", params={"fields": fields},
                                         json={"ids": [f"DOI:{d}" for d in dois]})
            if resp.status_code in {429, 500, 502, 503, 504}:
                ra = resp.headers.get("Retry-After")
                raise _Retryable(resp.status_code, float(ra) if ra and ra.isdigit() else None)
            if resp.status_code >= 400:
                raise SemanticScholarError(f"Semantic Scholar {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            if not isinstance(data, list) or len(data) != len(dois):
                raise SemanticScholarError("unexpected batch response shape")
            return data

        try:
            async for att in AsyncRetrying(
                retry=retry_if_exception(lambda e: isinstance(e, (_Retryable, httpx.TransportError))),
                wait=_wait, stop=stop_after_attempt(self.max_attempts), sleep=self._sleep, reraise=True,
            ):
                with att:
                    return await attempt()
        except _Retryable as e:
            hint = "" if self.has_key else " The keyless pool is shared and busy; set PAPERCUTS_S2_API_KEY or retry later."
            raise SemanticScholarError(f"Semantic Scholar kept returning {e.status}.{hint}") from e
        except httpx.TransportError as e:
            raise SemanticScholarError(f"Network error talking to Semantic Scholar: {e!r}") from e
        raise AssertionError("unreachable")  # pragma: no cover
