from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _doc_ids(ranking: Sequence[object]) -> list[str]:
    output: list[str] = []
    for item in ranking:
        if isinstance(item, str):
            output.append(item)
        elif hasattr(item, "doc_id"):
            output.append(str(getattr(item, "doc_id")))
        else:
            raise TypeError("Ranking items must be document IDs or objects with a doc_id attribute")
    return output


def precision_at_k(ranking: Sequence[object], relevant: set[str], k: int | None = None) -> float:
    docs = _doc_ids(ranking)
    if k is not None:
        if k < 0:
            raise ValueError("k must be >= 0 or None")
        docs = docs[:k]
    if not docs:
        return 0.0
    hits = sum(doc_id in relevant for doc_id in docs)
    return hits / len(docs)


def recall_at_k(ranking: Sequence[object], relevant: set[str], k: int | None = None) -> float:
    if not relevant:
        return 0.0
    docs = _doc_ids(ranking)
    if k is not None:
        if k < 0:
            raise ValueError("k must be >= 0 or None")
        docs = docs[:k]
    hits = sum(doc_id in relevant for doc_id in docs)
    return hits / len(relevant)


def average_precision(ranking: Sequence[object], relevant: set[str], k: int | None = None) -> float:
    if not relevant:
        return 0.0

    docs = _doc_ids(ranking)
    if k is not None:
        if k < 0:
            raise ValueError("k must be >= 0 or None")
        docs = docs[:k]

    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(docs, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant)


def mean_average_precision(
    rankings: Mapping[str, Sequence[object]],
    relevant_by_query: Mapping[str, set[str]],
    k: int | None = None,
) -> float:
    if not rankings:
        return 0.0
    scores = [
        average_precision(ranking, relevant_by_query.get(query_id, set()), k)
        for query_id, ranking in rankings.items()
    ]
    return sum(scores) / len(scores)


def dcg_at_k(
    ranking: Sequence[object],
    relevance: Mapping[str, float],
    k: int | None = None,
    *,
    exponential_gain: bool = True,
) -> float:
    """Compute standard DCG with log2(rank + 1) discount."""

    docs = _doc_ids(ranking)
    if k is not None:
        if k < 0:
            raise ValueError("k must be >= 0 or None")
        docs = docs[:k]

    dcg = 0.0
    for rank, doc_id in enumerate(docs, start=1):
        rel = float(relevance.get(doc_id, 0.0))
        gain = (2.0**rel - 1.0) if exponential_gain else rel
        dcg += gain / math.log2(rank + 1)
    return dcg


def ndcg_at_k(
    ranking: Sequence[object],
    relevance: Mapping[str, float],
    k: int | None = None,
    *,
    exponential_gain: bool = True,
) -> float:
    docs = _doc_ids(ranking)
    cutoff = len(docs) if k is None else min(k, len(docs))
    if cutoff <= 0:
        return 0.0

    actual = dcg_at_k(docs, relevance, cutoff, exponential_gain=exponential_gain)

    ideal_docs = [
        doc_id
        for doc_id, _ in sorted(
            relevance.items(),
            key=lambda item: (-item[1], item[0]),
        )[:cutoff]
    ]
    ideal = dcg_at_k(ideal_docs, relevance, cutoff, exponential_gain=exponential_gain)

    return 0.0 if ideal == 0.0 else actual / ideal
