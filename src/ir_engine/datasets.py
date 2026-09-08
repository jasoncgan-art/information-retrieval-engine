from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CranfieldData:
    documents: dict[str, str]
    queries: dict[str, str]
    binary_relevance: dict[str, set[str]]
    graded_relevance: dict[str, dict[str, float]]


def load_cranfield() -> CranfieldData:
    """Load the public Cranfield benchmark through ir_datasets.

    Raw benchmark files are intentionally not committed to this repository.
    ir_datasets downloads/caches the dataset according to its own data policy.
    """

    try:
        import ir_datasets
    except ImportError as exc:
        raise RuntimeError("Install project dependencies first: pip install -e .") from exc

    dataset = ir_datasets.load("cranfield")

    documents: dict[str, str] = {}
    for doc in dataset.docs_iter():
        text_parts = []
        for attribute in ("title", "text"):
            value = getattr(doc, attribute, None)
            if value:
                text_parts.append(str(value))
        documents[str(doc.doc_id)] = "\n".join(text_parts)

    queries = {
        str(query.query_id): str(query.text)
        for query in dataset.queries_iter()
    }

    binary_relevance: dict[str, set[str]] = {query_id: set() for query_id in queries}
    graded_relevance: dict[str, dict[str, float]] = {query_id: {} for query_id in queries}

    for qrel in dataset.qrels_iter():
        query_id = str(qrel.query_id)
        doc_id = str(qrel.doc_id)
        grade = float(qrel.relevance)
        graded_relevance.setdefault(query_id, {})[doc_id] = max(0.0, grade)
        if grade > 0:
            binary_relevance.setdefault(query_id, set()).add(doc_id)

    return CranfieldData(
        documents=documents,
        queries=queries,
        binary_relevance=binary_relevance,
        graded_relevance=graded_relevance,
    )
