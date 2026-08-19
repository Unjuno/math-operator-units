import torch

from opfusion import fusion_unseen_source_residual_gating as residual


def _scorer(vocab=31):
    return residual.IndependentResidualGate(
        vocabulary_size=vocab,
        sketch_size=6,
        hidden_size=10,
        sketch_seed=17,
        max_gate=4.0,
    )


def test_existing_gates_are_invariant_when_new_source_is_appended():
    torch.manual_seed(3)
    scorer = _scorer()
    logits4 = torch.randn(4, 31)
    logits5 = torch.cat([logits4, torch.randn(1, 31)], dim=0)
    norm = torch.stack([scorer.features_from_logits(logits4), scorer.features_from_logits(logits4 + 0.1)])
    scorer.set_normalization(norm)
    gates4 = scorer.gates_from_logits(logits4)
    gates5 = scorer.gates_from_logits(logits5)
    assert torch.allclose(gates4, gates5[:3], atol=1e-6, rtol=1e-6)


def test_residual_fusion_has_exact_zero_gate_identity():
    torch.manual_seed(5)
    logits4 = torch.randn(4, 29)
    gates3 = torch.rand(3)
    fused4 = residual.fuse_residual_logits(logits4, gates3)
    logits5 = torch.cat([logits4, torch.randn(1, 29)], dim=0)
    gates4 = torch.cat([gates3, torch.zeros(1)])
    fused5 = residual.fuse_residual_logits(logits5, gates4)
    assert torch.equal(fused4, fused5)


def test_specialist_permutation_equivariance():
    torch.manual_seed(7)
    scorer = _scorer(37)
    logits = torch.randn(5, 37)
    norm = torch.stack([scorer.features_from_logits(logits), scorer.features_from_logits(logits + 0.05)])
    scorer.set_normalization(norm)
    gates = scorer.gates_from_logits(logits)
    permutation = torch.tensor([0, 3, 1, 4, 2])
    permuted = scorer.gates_from_logits(logits.index_select(0, permutation))
    expected = gates.index_select(0, torch.tensor([2, 0, 3, 1]))
    assert torch.allclose(permuted, expected, atol=1e-6, rtol=1e-6)


def test_batched_residual_gate_training_has_finite_loss():
    torch.manual_seed(11)
    scorer = _scorer(23)
    logits = torch.randn(16, 4, 23)
    features = torch.stack([scorer.features_from_logits(row) for row in logits])
    targets = torch.randint(0, 23, (16,))
    batch = residual.ResidualTrainingBatch(features=features, source_logits=logits, target_ids=targets)
    report = residual.fit_residual_gate(
        scorer,
        batch,
        learning_rate=0.01,
        steps=4,
        batch_positions=8,
        gate_penalty=0.01,
        seed=13,
        device=torch.device('cpu'),
    )
    assert report['positions'] == 16
    assert report['nll_last'] == report['nll_last']
    assert 0.0 <= report['mean_gate_last'] <= 4.0
