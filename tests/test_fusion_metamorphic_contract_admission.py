from opfusion import fusion_metamorphic_contract_admission as admission
from opfusion import fusion_unseen_source_behavioral_admission as behavioral


def _mode(involved: int, neither: int):
    return {
        "subsets": {
            "heldout_involved": {"end_to_end_correct": involved},
            "heldout_neither": {"end_to_end_correct": neither},
        }
    }


def test_delta_and_marginal_use_correct_case_counts():
    baseline = _mode(10, 20)
    useful = _mode(13, 18)
    mixed = _mode(12, 17)
    assert admission._delta(useful, baseline) == {
        "heldout_involved": 3,
        "heldout_neither": -2,
        "net": 1,
    }
    assert admission._marginal(mixed, useful) == {
        "heldout_involved": -1,
        "heldout_neither": -1,
        "net": -2,
    }


def test_behavioral_score_patch_is_scoped():
    original = behavioral.behavioral_consensus
    with admission.patched_behavioral_score():
        assert behavioral.behavioral_consensus is admission.metamorphic_behavioral_score
    assert behavioral.behavioral_consensus is original


def test_default_calibration_seed_is_separate_from_prior_diagnostics():
    assert admission.DEFAULT_CALIBRATION_SEED == 749000
