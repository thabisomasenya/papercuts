"""Local text embeddings for papers, cached in SQLite.

Each paper is embedded once per model from its title + abstract. A hash of that
text is stored next to the vector, so `embed` only re-embeds papers whose text
changed (e.g. after an abstract backfill) or that are new.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

from papercuts.storage.db import Database


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    query_prefix: str = ""
    doc_sep: str = ". "  # between title and abstract


MODELS: dict[str, ModelSpec] = {
    # Fast, small (≈130 MB), strong general retrieval model. Default.
    "bge-small": ModelSpec("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
    # Better quality, ≈440 MB, slower.
    "bge-base": ModelSpec("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: "),
    # Trained on citation links between scientific papers; good for "papers like this one".
    "specter": ModelSpec("sentence-transformers/allenai-specter", "", "[SEP]"),
}


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers. Downloads the model on first use (cached in ~/.cache/huggingface)."""

    def __init__(self, name: str = "bge-small", device: str | None = None, batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "Semantic search needs the optional dependencies: pip install -e \".[semantic]\""
            ) from e
        self.spec = MODELS.get(name, ModelSpec(name))
        self.name = name
        self.batch_size = batch_size
        self._model = SentenceTransformer(self.spec.repo, device=device)
        # sentence-transformers ≥6 renamed get_sentence_embedding_dimension → get_embedding_dimension
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.spec.query_prefix + text])[0]


def paper_text(title: str, abstract: str | None, sep: str = ". ") -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    return f"{title}{sep}{abstract}" if abstract else title


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class EmbedSummary:
    model: str
    total: int
    embedded: int
    skipped_unchanged: int


def embed_corpus(
    db: Database,
    embedder: Embedder,
    *,
    batch_size: int = 64,
    force: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> EmbedSummary:
    sep = getattr(getattr(embedder, "spec", None), "doc_sep", ". ")
    rows = db.conn.execute(
        "SELECT p.id, p.title, p.abstract, e.text_hash FROM papers p "
        "LEFT JOIN embeddings e ON e.paper_id = p.id AND e.model = ?",
        (embedder.name,),
    ).fetchall()
    todo: list[tuple[str, str, str]] = []
    for r in rows:
        text = paper_text(r["title"], r["abstract"], sep)
        h = _hash(text)
        if force or r["text_hash"] != h:
            todo.append((r["id"], text, h))

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    done = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        vecs = embedder.embed_documents([t for _, t, _ in batch])
        with db.conn:
            db.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (paper_id, model, dim, text_hash, vector, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(pid, embedder.name, vecs.shape[1], h, vecs[i].astype(np.float32).tobytes(), now)
                 for i, (pid, _, h) in enumerate(batch)],
            )
        done += len(batch)
        if on_progress:
            on_progress(done, len(todo))
    return EmbedSummary(embedder.name, len(rows), done, len(rows) - done)


class VectorIndex:
    """All vectors for one model in a matrix. Brute-force cosine is exact and fast below ~100k papers."""

    def __init__(self, ids: list[str], matrix: np.ndarray, model: str):
        self.ids = ids
        self.matrix = matrix
        self.model = model
        self._pos = {pid: i for i, pid in enumerate(ids)}

    @classmethod
    def load(cls, db: Database, model: str) -> "VectorIndex":
        rows = db.conn.execute(
            "SELECT paper_id, dim, vector FROM embeddings WHERE model = ? ORDER BY paper_id", (model,)
        ).fetchall()
        if not rows:
            return cls([], np.zeros((0, 0), dtype=np.float32), model)
        dim = rows[0]["dim"]
        matrix = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(len(rows), dim)
        return cls([r["paper_id"] for r in rows], matrix, model)

    def __len__(self) -> int:
        return len(self.ids)

    def vector(self, paper_id: str) -> np.ndarray | None:
        i = self._pos.get(paper_id)
        return None if i is None else self.matrix[i]

    def search(
        self, query_vec: np.ndarray, k: int = 20, allowed: set[str] | None = None, exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        scores = self.matrix @ query_vec.astype(np.float32)
        if allowed is not None or exclude:
            mask = np.ones(len(self.ids), dtype=bool)
            if allowed is not None:
                mask &= np.fromiter((pid in allowed for pid in self.ids), bool, len(self.ids))
            if exclude:
                mask &= np.fromiter((pid not in exclude for pid in self.ids), bool, len(self.ids))
            scores = np.where(mask, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]
