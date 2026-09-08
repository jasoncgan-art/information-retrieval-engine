from __future__ import annotations

from dataclasses import dataclass

from .index import PositionalInvertedIndex


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float


def _numeric_suffix(identifier: str) -> tuple[int, str]:
    digits = "".join(ch for ch in identifier if ch.isdigit())
    return (int(digits), identifier) if digits else (10**18, identifier)


class TermFrequencyRanker:
    """Simple lexical baseline that sums query-term frequencies per document."""

    def __init__(self, index: PositionalInvertedIndex) -> None:
        self.index = index

    def search(self, query_tokens: list[str], *, top_k: int | None = None) -> list[SearchResult]:
        scores: dict[str, float] = {}

        for term in query_tokens:
            for doc_id, posting in self.index.postings(term).items():
                scores[doc_id] = scores.get(doc_id, 0.0) + posting.term_frequency

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], _numeric_suffix(item[0])),
        )

        if top_k is not None:
            if top_k < 0:
                raise ValueError("top_k must be >= 0 or None")
            ranked = ranked[:top_k]

        return [SearchResult(doc_id=doc_id, score=score) for doc_id, score in ranked]

    def search_many(
        self,
        queries: dict[str, list[str]],
        *,
        top_k: int | None = None,
    ) -> dict[str, list[SearchResult]]:
        return {
            query_id: self.search(tokens, top_k=top_k)
            for query_id, tokens in queries.items()
        }
