"""Small, transparent information-retrieval engine."""

from .index import PositionalInvertedIndex
from .preprocessing import TextPreprocessor
from .retrieval import SearchResult, TermFrequencyRanker

__all__ = [
    "PositionalInvertedIndex",
    "TextPreprocessor",
    "SearchResult",
    "TermFrequencyRanker",
]
