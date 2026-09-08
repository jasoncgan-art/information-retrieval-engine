from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


_TOKEN_RE = re.compile(r"\w+(?:[.-]\w+)*|[^\w\s]", flags=re.UNICODE)


@dataclass
class TextPreprocessor:
    """Normalize text for lexical retrieval.

    The pipeline lowercases text, tokenizes it, removes English stop words,
    and applies Porter stemming. Stop words can be injected to make the
    component easy to test without downloading NLTK resources.
    """

    stop_words: set[str] | None = None
    stemmer: PorterStemmer = field(default_factory=PorterStemmer)

    def __post_init__(self) -> None:
        if self.stop_words is None:
            try:
                self.stop_words = set(stopwords.words("english"))
            except LookupError as exc:
                raise RuntimeError(
                    "NLTK stopwords are missing. Run: python -m nltk.downloader stopwords"
                ) from exc

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def process(self, text: str, *, drop_final_period: bool = False) -> list[str]:
        tokens = [
            self.stemmer.stem(token)
            for token in self.tokenize(text)
            if token not in self.stop_words
        ]
        if drop_final_period and tokens and tokens[-1] == ".":
            tokens.pop()
        return tokens

    def process_many(self, texts: Iterable[str]) -> list[list[str]]:
        return [self.process(text) for text in texts]
