from opfusion import fusion_cross_source_behavioral_consensus as cross


def test_modal_output_handles_consensus_and_ties():
    assert cross.modal_output([3, 3, 4]) == (3, 2 / 3)
    assert cross.modal_output([3, 4]) == (None, 0.5)
    assert cross.modal_output([None, None]) == (None, 0.0)


def test_peer_scores_reward_shared_modal_output():
    rows = [
        {"source": "a", "modal_value": 7, "consensus": 1.0, "majority_correct": 1},
        {"source": "b", "modal_value": 7, "consensus": 1.0, "majority_correct": 1},
        {"source": "c", "modal_value": 2, "consensus": 1.0, "majority_correct": 0},
    ]
    scored = cross.attach_peer_scores(rows)
    assert scored[0]["peer_support"] == 0.5
    assert scored[1]["peer_support"] == 0.5
    assert scored[2]["peer_support"] == 0.0
    assert scored[0]["consensus_x_peer"] > scored[2]["consensus_x_peer"]


def test_peer_scores_are_source_order_equivariant():
    rows = [
        {"source": "a", "modal_value": 5, "consensus": 1.0, "majority_correct": 1},
        {"source": "b", "modal_value": 5, "consensus": 0.5, "majority_correct": 1},
        {"source": "c", "modal_value": 9, "consensus": 1.0, "majority_correct": 0},
        {"source": "d", "modal_value": None, "consensus": 0.5, "majority_correct": 0},
    ]
    original = {row["source"]: row for row in cross.attach_peer_scores(rows)}
    reordered = {row["source"]: row for row in cross.attach_peer_scores(list(reversed(rows)))}
    for source in original:
        assert original[source]["peer_support"] == reordered[source]["peer_support"]
        assert original[source]["peer_support_weighted"] == reordered[source]["peer_support_weighted"]


def test_score_summary_reports_stable_wrong_subset():
    rows = cross.attach_peer_scores(
        [
            {"source": "a", "modal_value": 7, "consensus": 1.0, "majority_correct": 1},
            {"source": "b", "modal_value": 7, "consensus": 1.0, "majority_correct": 1},
            {"source": "c", "modal_value": 2, "consensus": 1.0, "majority_correct": 0},
            {"source": "d", "modal_value": 8, "consensus": 0.5, "majority_correct": 0},
        ]
    )
    summary = cross.score_summary(rows)
    assert summary["consensus_one_rows"] == 3
    assert summary["consensus_one_positive_rows"] == 2
    assert summary["consensus_one_negative_rows"] == 1
    assert summary["scores"]["consensus_x_peer"]["auc_consensus_one"] == 1.0
