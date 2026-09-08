# Information Retrieval Engine

A small lexical search engine built from scratch to make the mechanics of information retrieval transparent: text normalization, a **positional inverted index**, document ranking, and evaluation with standard IR metrics.

The project uses the classic **Cranfield** benchmark (1,400 scientific documents and 225 queries) through `ir_datasets`; raw benchmark files are not committed to the repository.

## Why this project

Modern search stacks often hide indexing and ranking behind libraries or services. This project implements the core retrieval loop directly so the data structures and ranking tradeoffs are inspectable end to end.

**What is implemented**

- text tokenization, stop-word removal, and Porter stemming
- positional inverted index: `term -> document -> (frequency, positions)`
- document-frequency pruning with configurable `min_df`
- term-frequency lexical ranking with deterministic tie-breaking
- Precision@K and Recall@K
- Average Precision and Mean Average Precision (MAP)
- DCG / NDCG with graded relevance
- Cranfield benchmark runner and `min_df` experiment script
- unit tests for indexing, ranking, and evaluation behavior

## Architecture

```text
raw text
   |
   v
TextPreprocessor
   |
   v
PositionalInvertedIndex
   |
   v
TermFrequencyRanker
   |
   +--------------------+
   v                    v
ranked results       relevance judgments
   |                    |
   +---------+----------+
             v
        IR evaluation
   Precision / Recall / MAP / NDCG
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
python -m nltk.downloader stopwords
pytest
python examples/quickstart.py
```

Run the public Cranfield benchmark:

```bash
python -m ir_engine.cli --min-df 10 --top-k 10
```

Compare vocabulary-pruning choices:

```bash
python scripts/benchmark.py
```

`ir_datasets` downloads/caches Cranfield when needed. See [`data/README.md`](data/README.md).

## Retrieval strategy

For every processed query term, the ranker reads its posting list and accumulates the term frequency for each matching document:

```text
score(document, query) = sum TF(term, document)
                         for terms present in the query
```

Documents are sorted by descending score. Numeric document ID is used as a deterministic tie-breaker.

This is intentionally a simple lexical baseline. It makes a good reference point for stronger ranking methods such as TF-IDF and BM25.

## Evaluation

The project separates binary relevance metrics from graded relevance metrics:

- **Precision@K** — how much of the returned top K is relevant?
- **Recall@K** — how much of all known relevant material was recovered?
- **Average Precision / MAP** — rewards systems that place relevant documents earlier in the ranking.
- **NDCG@K** — rewards putting *more highly relevant* documents earlier, with logarithmic rank discounting.

For NDCG, the implementation uses the common formulation `(2^relevance - 1) / log2(rank + 1)`.

## Design choices and tradeoffs

### Positional index instead of a simple posting set
Keeping token positions costs more memory, but creates a path toward phrase queries, proximity scoring, and highlighting without rebuilding the index.

### `min_df` vocabulary pruning
Rare terms can be noisy and increase index size. The benchmark script makes this tradeoff measurable rather than hard-coding one threshold.

### Transparent TF baseline
The current ranker is deliberately explainable. A natural next step is to implement BM25 and compare it against the TF baseline using the same evaluation harness.

## Next improvements

1. Add TF-IDF and BM25 rankers and compare MAP/NDCG.
2. Add phrase and proximity queries using stored token positions.
3. Add query-term weighting and document-length normalization.
4. Add a dense embedding retriever and evaluate lexical vs. semantic vs. hybrid search.
5. Profile index build time, index size, and query latency.

## Interview talking points

- Why inverted indexes make lexical search efficient.
- Difference between term frequency and document frequency.
- Why `min_df` changes both vocabulary size and retrieval quality.
- Why Precision and Recall can tell different stories about the same result set.
- Why MAP is sensitive to the position of relevant results.
- Why NDCG is useful when relevance is graded rather than binary.
- How this baseline could evolve into BM25 or hybrid retrieval for RAG.

## Dataset

Cranfield is a classic information-retrieval test collection. This repository **does not redistribute raw course files or a course-provided dataset copy**. The benchmark is loaded through the public `ir_datasets` interface, and the original collection is available from the University of Glasgow Information Retrieval Group.
