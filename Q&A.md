## Project summary

I built a small information-retrieval engine from scratch rather than relying on Elasticsearch or a vector database. I implemented text preprocessing, a positional inverted index, term-frequency ranking, and an evaluation harness with Precision/Recall, MAP, and NDCG. I used the Cranfield benchmark so I could quantify ranking quality, and I added an experiment to study the tradeoff between vocabulary pruning and retrieval effectiveness.

## Why positional postings?

A basic inverted index only needs to know which documents contain a term. I also keep term frequency and positions. Frequency supports ranking; positions enable future phrase/proximity queries and snippet highlighting.

## What would you improve first?

BM25. The current TF ranker is a deliberately simple baseline. BM25 adds document-length normalization and saturating term-frequency weighting, so it should provide a meaningful measurable improvement without changing the index architecture.

## How does this relate to RAG?

RAG still needs a retrieval system. Dense embeddings are useful for semantic similarity, but lexical retrieval remains strong for exact entities, rare terms, numbers, and jargon. This project gives me a transparent lexical baseline that could be combined with dense retrieval in a hybrid system.

## What did you learn?

The main lesson is that retrieval quality is not one number. Precision, recall, MAP, and NDCG reward different behavior. Ranking decisions should be tied to the product objective and evaluated at the cutoff users actually see.
