from __future__ import annotations

import argparse

from .datasets import load_cranfield
from .evaluation import mean_average_precision, ndcg_at_k, precision_at_k, recall_at_k
from .index import PositionalInvertedIndex
from .preprocessing import TextPreprocessor
from .retrieval import TermFrequencyRanker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Cranfield lexical retrieval benchmark")
    parser.add_argument("--min-df", type=int, default=10, help="Minimum document frequency")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results per query")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    data = load_cranfield()
    preprocessor = TextPreprocessor()

    tokenized_docs = {
        doc_id: preprocessor.process(text)
        for doc_id, text in data.documents.items()
    }
    tokenized_queries = {
        query_id: preprocessor.process(text, drop_final_period=True)
        for query_id, text in data.queries.items()
    }

    index = PositionalInvertedIndex().build(tokenized_docs, min_df=args.min_df)
    ranker = TermFrequencyRanker(index)
    rankings = ranker.search_many(tokenized_queries, top_k=args.top_k)

    p_at_k = [
        precision_at_k(rankings[qid], data.binary_relevance[qid], args.top_k)
        for qid in rankings
    ]
    r_at_k = [
        recall_at_k(rankings[qid], data.binary_relevance[qid], args.top_k)
        for qid in rankings
    ]
    ndcg_scores = [
        ndcg_at_k(rankings[qid], data.graded_relevance[qid], args.top_k)
        for qid in rankings
    ]

    print(f"Documents: {len(data.documents):,}")
    print(f"Queries:   {len(data.queries):,}")
    print(f"Terms:     {len(index.terms):,} (min_df={args.min_df})")
    print(f"P@{args.top_k}:     {sum(p_at_k) / len(p_at_k):.4f}")
    print(f"R@{args.top_k}:     {sum(r_at_k) / len(r_at_k):.4f}")
    print(f"MAP@{args.top_k}:   {mean_average_precision(rankings, data.binary_relevance, args.top_k):.4f}")
    print(f"NDCG@{args.top_k}:  {sum(ndcg_scores) / len(ndcg_scores):.4f}")


if __name__ == "__main__":
    main()
