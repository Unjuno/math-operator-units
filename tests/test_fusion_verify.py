import torch

from opfusion.fusion_verify import (
    CandidateSpec,
    _candidate_sort_key,
    causal_rms_equalize_biases,
    fuse_next_logits,
)


def test_causal_rms_equalization_is_position_local_and_centered() -> None:
    biases = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-4.0, -2.0, 0.0, 2.0],
        ]
    )
    equalized = causal_rms_equalize_biases(biases)
    assert equalized.shape == biases.shape
    assert torch.allclose(equalized.mean(dim=-1), torch.zeros(2), atol=1e-6)
    rms = equalized.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms[0], rms[1], atol=1e-6)


def test_causal_equalization_keeps_zero_bias_zero() -> None:
    biases = torch.stack([torch.zeros(7), torch.arange(7, dtype=torch.float32)])
    equalized = causal_rms_equalize_biases(biases)
    assert torch.equal(equalized[0], torch.zeros(7))
    assert float(equalized[1].abs().sum()) > 0.0


def test_raw_next_logit_fusion_matches_declared_formula() -> None:
    base = torch.randn(11)
    left = torch.randn(11)
    right = torch.randn(11)
    fused = fuse_next_logits(base, [left, right], mode="raw_sum", alpha=0.5)
    expected = base + 0.5 * ((left - base) + (right - base))
    assert torch.allclose(fused, expected)


def test_singleton_causal_equalization_is_softmax_equivalent_at_alpha_one() -> None:
    base = torch.randn(13)
    specialist = torch.randn(13)
    fused = fuse_next_logits(
        base,
        [specialist],
        mode="causal_rms_equalized_sum",
        alpha=1.0,
    )
    assert torch.allclose(torch.softmax(fused, dim=-1), torch.softmax(specialist, dim=-1), atol=1e-6)


def test_candidate_operator_mask_uses_canonical_order() -> None:
    candidate = CandidateSpec("triple", 25, "raw_sum", 0.25)
    assert candidate.operators == ("scalar.add", "scalar.min", "scalar.max")


def test_passing_candidate_sorts_before_nonpassing_candidate() -> None:
    common = {
        "active_final_accuracy_min": 0.95,
        "active_final_accuracy_macro": 0.95,
        "active_trace_validity_min": 0.95,
        "inactive_base_exact_agreement_macro": 0.95,
        "active_response_exact_macro": 0.80,
        "operators": ["scalar.add"],
    }
    passing = {**common, "passes_validation_gate": True}
    failing = {**common, "passes_validation_gate": False}
    assert _candidate_sort_key(passing) < _candidate_sort_key(failing)
