import torch

from opfusion import fusion_unseen_source_decoy_calibration as decoy
from opfusion import fusion_unseen_source_residual_gating as residual


def _scorer(vocab=31):
    return residual.IndependentResidualGate(
        vocabulary_size=vocab,
        sketch_size=6,
        hidden_size=10,
        sketch_seed=17,
        max_gate=4.0,
    )


def test_batched_features_match_single_position_features():
    torch.manual_seed(3)
    scorer = _scorer(29)
    logits = torch.randn(7, 4, 29)
    batched = decoy.batched_base_relative_features(logits, scorer.projection)
    scalar = torch.stack([scorer.features_from_logits(row) for row in logits])
    assert torch.allclose(batched, scalar, atol=1e-6, rtol=1e-6)


def test_cross_prefix_decoy_uses_current_base_and_other_prefix_delta():
    torch.manual_seed(5)
    logits = torch.randn(8, 4, 23)
    decoys = decoy.build_cross_prefix_decoys(logits)
    assert decoys.shape == (8, 23)
    offset = 4
    donor_position = offset
    donor_specialist = 1
    donor_delta = logits[donor_position, donor_specialist] - logits[donor_position, 0]
    donor_delta = donor_delta - donor_delta.mean()
    expected = logits[0, 0] + donor_delta
    assert torch.allclose(decoys[0], expected, atol=1e-6, rtol=1e-6)


def test_decoy_batch_preserves_real_training_rows():
    torch.manual_seed(7)
    scorer = _scorer(19)
    sources = torch.randn(10, 4, 19)
    features = torch.stack([scorer.features_from_logits(row) for row in sources])
    targets = torch.randint(0, 19, (10,))
    base = residual.ResidualTrainingBatch(
        features=features,
        source_logits=sources,
        target_ids=targets,
    )
    batch = decoy.make_decoy_batch(base)
    assert torch.equal(batch.source_logits, sources)
    assert torch.equal(batch.target_ids, targets)
    assert batch.decoy_source_logits.shape == (10, 19)


def test_decoy_training_is_finite_and_tracks_null_gate():
    torch.manual_seed(11)
    scorer = _scorer(17)
    sources = torch.randn(20, 4, 17)
    features = torch.stack([scorer.features_from_logits(row) for row in sources])
    targets = torch.randint(0, 17, (20,))
    base = residual.ResidualTrainingBatch(
        features=features,
        source_logits=sources,
        target_ids=targets,
    )
    batch = decoy.make_decoy_batch(base)
    report = decoy.fit_with_decoys(
        scorer,
        batch,
        learning_rate=0.01,
        steps=5,
        batch_positions=10,
        real_gate_penalty=0.01,
        decoy_gate_penalty=0.1,
        seed=13,
        device=torch.device('cpu'),
    )
    assert report['positions'] == 20
    assert report['nll_last'] == report['nll_last']
    assert 0.0 <= report['decoy_gate_mean_last'] <= 4.0
    assert 0.0 <= report['real_gate_mean_last'] <= 4.0
