import torch

from opfusion import fusion_unseen_source_field_sketch as sketch


def test_field_sketch_features_are_finite():
    torch.manual_seed(3)
    scorer = sketch.FieldSketchSourceScorer(
        vocabulary_size=37,
        sketch_size=8,
        hidden_size=12,
        sketch_seed=5,
    )
    logits = torch.randn(5, 37)
    features = scorer.field_features(logits)
    assert features.shape == (5, len(sketch.summary_baseline.FEATURE_NAMES) + 4 * 8)
    assert torch.isfinite(features).all()


def test_field_sketch_scorer_is_equivariant_over_specialists():
    torch.manual_seed(7)
    scorer = sketch.FieldSketchSourceScorer(
        vocabulary_size=41,
        sketch_size=10,
        hidden_size=16,
        sketch_seed=11,
    )
    logits = torch.randn(5, 41)
    training_features = torch.stack(
        [scorer.field_features(logits), scorer.field_features(logits + 0.05)], dim=0
    )
    scorer.set_normalization(training_features)
    original = scorer.weights_from_logits(logits)
    permutation = torch.tensor([0, 3, 4, 1, 2])
    permuted = scorer.weights_from_logits(logits.index_select(0, permutation))
    assert torch.allclose(
        permuted,
        original.index_select(0, permutation),
        atol=1e-6,
        rtol=1e-6,
    )


def test_field_sketch_supports_new_source_count():
    torch.manual_seed(13)
    scorer = sketch.FieldSketchSourceScorer(
        vocabulary_size=31,
        sketch_size=6,
        hidden_size=10,
        sketch_seed=17,
    )
    train_logits = torch.randn(4, 31)
    train_features = torch.stack(
        [scorer.field_features(train_logits), scorer.field_features(train_logits + 0.1)], dim=0
    )
    scorer.set_normalization(train_features)
    eval_logits = torch.randn(5, 31)
    weights = scorer.weights_from_logits(eval_logits)
    assert weights.shape == (5,)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_field_scorer_nll_optimizes_on_synthetic_batch():
    torch.manual_seed(19)
    scorer = sketch.FieldSketchSourceScorer(
        vocabulary_size=23,
        sketch_size=4,
        hidden_size=8,
        sketch_seed=23,
    )
    features = torch.randn(20, 4, scorer.feature_size)
    target = torch.rand(20, 4).clamp_min(1e-4)
    batch = sketch.SketchTrainingBatch(
        features=features,
        target_source_probabilities=target,
    )
    report = sketch.fit_field_scorer(
        scorer,
        batch,
        learning_rate=0.01,
        steps=5,
        batch_positions=10,
        seed=29,
        device=torch.device("cpu"),
    )
    assert report["positions"] == 20
    assert report["source_count_train"] == 4
    assert report["feature_size"] == scorer.feature_size
    assert report["optimization_last"] == report["optimization_last"]
