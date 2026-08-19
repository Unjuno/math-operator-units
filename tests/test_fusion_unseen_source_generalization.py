import torch

from opfusion import fusion_unseen_source_generalization as unseen


def test_source_features_are_finite_and_identity_free_shape():
    torch.manual_seed(7)
    logits = torch.randn(5, 31)
    features = unseen._source_features(logits)
    assert features.shape == (5, len(unseen.FEATURE_NAMES))
    assert torch.isfinite(features).all()


def test_shared_scorer_is_equivariant_to_specialist_permutation():
    torch.manual_seed(11)
    logits = torch.randn(5, 29)
    scorer = unseen.SharedSourceScorer(hidden_size=8)
    training = torch.stack(
        [unseen._source_features(logits), unseen._source_features(logits + 0.1)], dim=0
    )
    scorer.set_normalization(training)
    original = scorer.weights_from_logits(logits)
    permutation = torch.tensor([0, 3, 1, 4, 2])
    permuted = scorer.weights_from_logits(logits.index_select(0, permutation))
    assert torch.allclose(
        permuted,
        original.index_select(0, permutation),
        atol=1e-6,
        rtol=1e-6,
    )


def test_shared_scorer_nll_has_gradients():
    torch.manual_seed(13)
    features = torch.randn(12, 4, len(unseen.FEATURE_NAMES))
    target = torch.rand(12, 4).clamp_min(1e-4)
    batch = unseen.TrainingBatch(features=features, target_source_probabilities=target)
    scorer, report = unseen.fit_shared_scorer(
        batch,
        hidden_size=8,
        learning_rate=0.01,
        steps=4,
        batch_positions=8,
        seed=17,
        device=torch.device("cpu"),
    )
    assert report["positions"] == 12
    assert report["source_count_train"] == 4
    assert report["optimization_last"] == report["optimization_last"]
    assert all(parameter.grad is not None for parameter in scorer.parameters())


def test_subset_summary_counts_expected_pairs():
    pairs = {}
    for inner in unseen.FUNCTIONAL_OPERATORS:
        for outer in unseen.FUNCTIONAL_OPERATORS:
            counter = unseen.sequential._empty_counter()
            counter["cases"] = 2
            pairs[f"{inner}->{outer}"] = unseen.sequential._finalize_counter(counter)
    heldout = unseen.FUNCTIONAL_OPERATORS[0]
    summary = unseen._subset_summary(pairs, heldout=heldout)
    assert summary["heldout_as_inner"]["cases"] == 8
    assert summary["heldout_as_outer"]["cases"] == 8
    assert summary["heldout_involved"]["cases"] == 14
    assert summary["heldout_neither"]["cases"] == 18
