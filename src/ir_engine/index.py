from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Posting:
    term_frequency: int
    positions: tuple[int, ...]


class PositionalInvertedIndex:
    """Map terms to documents, term frequencies, and token positions."""

    def __init__(self) -> None:
        self._index: dict[str, dict[str, Posting]] = {}
        self.document_count = 0

    @property
    def terms(self) -> set[str]:
        return set(self._index)

    def __contains__(self, term: str) -> bool:
        return term in self._index

    def postings(self, term: str) -> dict[str, Posting]:
        return self._index.get(term, {})

    def document_frequency(self, term: str) -> int:
        return len(self.postings(term))

    def build(
        self,
        documents: dict[str, list[str]],
        *,
        min_df: int = 1,
    ) -> "PositionalInvertedIndex":
        if min_df < 1:
            raise ValueError("min_df must be >= 1")

        mutable: dict[str, dict[str, list[object]]] = {}

        for doc_id, tokens in documents.items():
            for position, term in enumerate(tokens):
                term_docs = mutable.setdefault(term, {})
                frequency_positions = term_docs.setdefault(doc_id, [0, []])
                frequency_positions[0] += 1
                frequency_positions[1].append(position)

        self._index = {}
        for term, term_docs in mutable.items():
            if len(term_docs) < min_df:
                continue
            self._index[term] = {
                doc_id: Posting(
                    term_frequency=int(values[0]),
                    positions=tuple(values[1]),
                )
                for doc_id, values in term_docs.items()
            }

        self.document_count = len(documents)
        return self

    def as_dict(self) -> dict[str, dict[str, Posting]]:
        return self._index
