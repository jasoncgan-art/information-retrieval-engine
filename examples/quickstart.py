from ir_engine import PositionalInvertedIndex, TermFrequencyRanker, TextPreprocessor


DOCUMENTS = {
    "d1": "Information retrieval systems rank documents for a user query.",
    "d2": "An inverted index maps terms to documents and token positions.",
    "d3": "Search engines combine indexing, ranking, and evaluation.",
}


def main() -> None:
    preprocessor = TextPreprocessor(stop_words={"a", "an", "and", "for", "to"})
    tokenized_docs = {doc_id: preprocessor.process(text) for doc_id, text in DOCUMENTS.items()}

    index = PositionalInvertedIndex().build(tokenized_docs)
    ranker = TermFrequencyRanker(index)

    query = preprocessor.process("inverted index search")
    for result in ranker.search(query, top_k=3):
        print(f"{result.doc_id}: {result.score:.1f}")


if __name__ == "__main__":
    main()
