from ir_engine.index import PositionalInvertedIndex
from ir_engine.retrieval import TermFrequencyRanker


def test_positional_index_tracks_frequency_and_positions():
    docs = {
        "d1": ["data", "science", "data"],
        "d2": ["science", "search"],
    }
    index = PositionalInvertedIndex().build(docs)

    posting = index.postings("data")["d1"]
    assert posting.term_frequency == 2
    assert posting.positions == (0, 2)
    assert index.document_frequency("science") == 2


def test_min_df_prunes_rare_terms():
    docs = {
        "d1": ["common", "rare"],
        "d2": ["common"],
    }
    index = PositionalInvertedIndex().build(docs, min_df=2)

    assert "common" in index
    assert "rare" not in index


def test_ranker_sums_term_frequency_and_breaks_ties_by_doc_number():
    docs = {
        "d12": ["search", "search"],
        "d100": ["search", "search"],
        "d3": ["search", "search", "search"],
    }
    index = PositionalInvertedIndex().build(docs)
    results = TermFrequencyRanker(index).search(["search"])

    assert [result.doc_id for result in results] == ["d3", "d12", "d100"]
