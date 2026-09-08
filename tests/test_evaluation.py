import math

from ir_engine.evaluation import (
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_and_recall_at_k():
    ranking = ["d1", "d2", "d3"]
    relevant = {"d1", "d3", "d9"}

    assert math.isclose(precision_at_k(ranking, relevant, 2), 0.5)
    assert math.isclose(recall_at_k(ranking, relevant, 2), 1 / 3)


def test_average_precision_rewards_early_relevant_documents():
    relevant = {"d1", "d3"}
    assert math.isclose(average_precision(["d1", "d2", "d3"], relevant), (1 + 2 / 3) / 2)


def test_map_is_mean_of_query_average_precisions():
    rankings = {"q1": ["d1", "d2"], "q2": ["d2", "d1"]}
    relevant = {"q1": {"d1"}, "q2": {"d1"}}
    assert math.isclose(mean_average_precision(rankings, relevant), (1.0 + 0.5) / 2)


def test_ndcg_is_one_for_ideal_ranking():
    relevance = {"d1": 3, "d2": 2, "d3": 1}
    assert math.isclose(ndcg_at_k(["d1", "d2", "d3"], relevance, 3), 1.0)


def test_ndcg_penalizes_misordered_ranking():
    relevance = {"d1": 3, "d2": 2, "d3": 1}
    score = ndcg_at_k(["d3", "d2", "d1"], relevance, 3)
    assert 0.0 < score < 1.0
