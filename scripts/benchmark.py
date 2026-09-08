"""Compare vocabulary pruning choices on the public Cranfield benchmark."""

from ir_engine.datasets import load_cranfield
from ir_engine.evaluation import mean_average_precision, ndcg_at_k
from ir_engine.index import PositionalInvertedIndex
from ir_engine.preprocessing import TextPreprocessor
from ir_engine.retrieval import TermFrequencyRanker


def main() -> None:
    data = load_cranfield()
    preprocessor = TextPreprocessor()

    documents = {doc_id: preprocessor.process(text) for doc_id, text in data.documents.items()}
    queries = {
        query_id: preprocessor.process(text, drop_final_period=True)
        for query_id, text in data.queries.items()
    }

    print("min_df\tterms\tMAP@10\tNDCG@10")
    for min_df in (1, 5, 10, 20):
        index = PositionalInvertedIndex().build(documents, min_df=min_df)
        rankings = TermFrequencyRanker(index).search_many(queries, top_k=10)
        map_10 = mean_average_precision(rankings, data.binary_relevance, 10)
        ndcg_10 = sum(
            ndcg_at_k(rankings[qid], data.graded_relevance[qid], 10)
            for qid in rankings
        ) / len(rankings)
        print(f"{min_df}\t{len(index.terms)}\t{map_10:.4f}\t{ndcg_10:.4f}")


if __name__ == "__main__":
    main()
