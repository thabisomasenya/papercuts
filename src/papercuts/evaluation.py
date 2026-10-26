"""Search evaluation: build relevance judgments by pooling, then score methods with nDCG / recall / MRR.

Judgments live in a JSON-lines file (default eval/qrels.jsonl), one query per line:
    {"query": "...", "judgments": {"openalex:W123": 2, "openalex:W456": 0}}
Grades: 0 = not relevant, 1 = partly relevant, 2 = highly relevant.

Pooling (as in TREC): for each query, the top results of every method are merged
and shown in random order with the method hidden, so judging isn't biased towards
any one method. Results nobody judged count as not relevant, which is fair as long
as the pool depth covers the cut-off being measured.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SearchFn = Callable[[str, int], list[str]]  # (query, k) -> ranked paper ids


# ----------------------------------------------------------------------------- qrels file


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, int]] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            out[rec["query"]] = {pid: int(g) for pid, g in rec.get("judgments", {}).items()}
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"{path}:{n}: bad line ({e})") from e
    return out


def save_qrels(path: Path, qrels: dict[str, dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps({"query": q, "judgments": j}, ensure_ascii=False) + "\n" for q, j in qrels.items()))
    tmp.replace(path)  # atomic: a crash mid-write never corrupts your judgments


def pool(methods: dict[str, SearchFn], query: str, depth: int = 10, seed: int = 0) -> list[str]:
    seen: dict[str, None] = {}
    for fn in methods.values():
        for pid in fn(query, depth):
            seen.setdefault(pid, None)
    ids = list(seen)
    random.Random(f"{seed}:{query}").shuffle(ids)
    return ids


# ----------------------------------------------------------------------------- metrics


def dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: list[str], judgments: dict[str, int], k: int = 10) -> float:
    ideal = dcg(sorted(judgments.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([judgments.get(pid, 0) for pid in ranked[:k]]) / ideal


def recall_at_k(ranked: list[str], judgments: dict[str, int], k: int = 20) -> float:
    relevant = {pid for pid, g in judgments.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def mrr(ranked: list[str], judgments: dict[str, int]) -> float:
    for i, pid in enumerate(ranked, 1):
        if judgments.get(pid, 0) > 0:
            return 1.0 / i
    return 0.0


@dataclass
class MethodScore:
    method: str
    ndcg10: float
    recall20: float
    mrr: float
    n_queries: int


def evaluate(methods: dict[str, SearchFn], qrels: dict[str, dict[str, int]]) -> list[MethodScore]:
    usable = {q: j for q, j in qrels.items() if any(g > 0 for g in j.values())}
    scores = []
    for name, fn in methods.items():
        n = r = m = 0.0
        for q, j in usable.items():
            ranked = fn(q, 20)
            n += ndcg_at_k(ranked, j, 10)
            r += recall_at_k(ranked, j, 20)
            m += mrr(ranked, j)
        k = max(len(usable), 1)
        scores.append(MethodScore(name, n / k, r / k, m / k, len(usable)))
    return scores


def results_markdown(scores: list[MethodScore], n_judged: int) -> str:
    best = max(scores, key=lambda s: s.ndcg10).method if scores else ""
    lines = [
        f"Evaluated on {scores[0].n_queries if scores else 0} queries ({n_judged} relevance judgments).",
        "",
        "| Method | nDCG@10 | Recall@20 | MRR |",
        "|---|---|---|---|",
    ]
    for s in scores:
        name = f"**{s.method}**" if s.method == best else s.method
        lines.append(f"| {name} | {s.ndcg10:.3f} | {s.recall20:.3f} | {s.mrr:.3f} |")
    return "\n".join(lines) + "\n"
