from opfusion import fusion_hard_null_plugin_admission as hard


def test_choose_hard_threshold_prefers_separating_cut():
    rows = [
        {'consensus': 1.0, 'majority_correct': 1},
        {'consensus': 1.0, 'majority_correct': 1},
        {'consensus': 2 / 3, 'majority_correct': 0},
        {'consensus': 1 / 3, 'majority_correct': 0},
    ]
    report = hard.choose_hard_threshold(rows)
    assert 2 / 3 < report['threshold'] <= 1.0
    assert report['balanced_accuracy'] == 1.0
    assert report['fp'] == 0
    assert report['fn'] == 0


def test_choose_threshold_conservative_tie_breaks_upward():
    rows = [
        {'consensus': 1.0, 'majority_correct': 1},
        {'consensus': 1.0, 'majority_correct': 0},
    ]
    report = hard.choose_hard_threshold(rows)
    assert report['threshold'] > 1.0
    assert report['balanced_accuracy'] == 0.5


def test_hard_admission_maps_subthreshold_exactly_to_identity_factor():
    assert hard.hard_admission_factor(0.66, power=2.0, threshold=0.75) == 0.0
    assert hard.hard_admission_factor(1.0, power=2.0, threshold=0.75) == 1.0


def test_delta_and_marginal_use_disjoint_subset_counts():
    baseline = {
        'subsets': {
            'heldout_involved': {'end_to_end_correct': 10},
            'heldout_neither': {'end_to_end_correct': 20},
        }
    }
    useful = {
        'subsets': {
            'heldout_involved': {'end_to_end_correct': 14},
            'heldout_neither': {'end_to_end_correct': 19},
        }
    }
    mixed = {
        'subsets': {
            'heldout_involved': {'end_to_end_correct': 13},
            'heldout_neither': {'end_to_end_correct': 19},
        }
    }
    assert hard._delta(useful, baseline) == {
        'heldout_involved': 4,
        'heldout_neither': -1,
        'net': 3,
    }
    assert hard._marginal(mixed, useful) == {
        'heldout_involved': -1,
        'heldout_neither': 0,
        'net': -1,
    }
