"""Settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader so we don't need python-dotenv. Existing env vars win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    db_path: Path
    openalex_api_key: str | None
    mailto: str | None
    max_concurrency: int
    request_timeout: float

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            db_path=Path(os.getenv("PAPERCUTS_DB", "data/papercuts.db")),
            openalex_api_key=os.getenv("PAPERCUTS_OPENALEX_API_KEY") or None,
            mailto=os.getenv("PAPERCUTS_MAILTO") or None,
            max_concurrency=int(os.getenv("PAPERCUTS_MAX_CONCURRENCY", "4")),
            request_timeout=float(os.getenv("PAPERCUTS_TIMEOUT", "30")),
        )
