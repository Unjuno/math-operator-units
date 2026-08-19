from opfusion import fusion_unseen_source_rollout_consensus as consensus


def test_equivalent_views_are_deterministic_permutations():
    views = consensus.equivalent_views((1, 2, 3))
    assert views == ((1, 2, 3), (3, 2, 1), (2, 3, 1))
    assert all(sorted(view) == [1, 2, 3] for view in views)
    assert consensus.equivalent_views((4, 7)) == ((4, 7), (7, 4))


def test_summarize_view_outputs_counts_parse_consensus_and_correctness():
    row = consensus.summarize_view_outputs([9, 9, None], expected=9)
    assert row['views'] == 3
    assert row['parse_count'] == 2
    assert row['parse_rate'] == 2 / 3
    assert row['consensus'] == 2 / 3
    assert row['correct_fraction'] == 2 / 3
    assert row['majority_correct'] == 1
    assert row['modal_value'] == 9


def test_pairwise_auc_handles_ties():
    rows = [
        {'score': 1.0, 'label': 1},
        {'score': 0.5, 'label': 1},
        {'score': 0.5, 'label': 0},
        {'score': 0.0, 'label': 0},
    ]
    assert consensus.pairwise_auc(rows, score_key='score', label_key='label') == 0.875


def test_example_selection_is_tie_aware():
    rows = [
        {
            'cohort_id': 'c0', 'task_operator': 'scalar.add', 'sample_index': 0,
            'source': 'scalar.add', 'consensus': 1.0, 'majority_correct': 1,
            'correct_fraction': 1.0,
        },
        {
            'cohort_id': 'c0', 'task_operator': 'scalar.add', 'sample_index': 0,
            'source': 'Base', 'consensus': 1.0, 'majority_correct': 0,
            'correct_fraction': 0.0,
        },
        {
            'cohort_id': 'c0', 'task_operator': 'scalar.add', 'sample_index': 0,
            'source': 'scalar.max', 'consensus': 0.5, 'majority_correct': 0,
            'correct_fraction': 0.0,
        },
    ]
    report = consensus.summarize_example_selection(rows)
    assert report['examples'] == 1
    assert report['top_consensus_set_contains_majority_correct_source_rate'] == 1.0
    assert report['all_top_consensus_sources_majority_correct_rate'] == 0.0
    assert report['matching_specialist_in_top_consensus_set_rate'] == 1.0
    assert report['matching_specialist_strictly_highest_consensus_rate'] == 0.0
