import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fraud.extensions import CONTINUOUS, nested_subsets, perturbed_inputs
from fraud.scratch_logistic import ScratchLogisticRegression, objective, sigmoid


def test_scratch_matches_sklearn_for_l2_and_free_intercept():
    rng = np.random.default_rng(16)
    features = rng.normal(size=(800, 7))
    labels = rng.binomial(1, sigmoid(-1.8 + features[:, 0] - 0.5 * features[:, 2]))
    design = np.column_stack([features, np.ones(len(features))])
    c = 0.4
    actual = ScratchLogisticRegression(c=c).fit(features, labels)
    expected = LogisticRegression(C=c, solver="lbfgs", tol=1e-12).fit(features, labels)
    strength = 1 / (c * len(labels))
    expected_loss = objective(design, labels, np.r_[expected.coef_[0], expected.intercept_[0]], strength)
    actual_loss = objective(design, labels, np.r_[actual.coef_, actual.intercept_], strength)
    assert actual_loss == pytest.approx(expected_loss, abs=1e-11)
    np.testing.assert_allclose(actual.predict_proba(features), expected.predict_proba(features), atol=1e-5)
    assert actual.gradient_norm_ <= actual.tol


def test_scratch_handles_extreme_logits_and_unpenalized_intercept():
    np.testing.assert_array_equal(sigmoid(np.array([-1000.0, 0.0, 1000.0])), [0.0, 0.5, 1.0])
    labels = np.r_[np.zeros(9), 1]
    model = ScratchLogisticRegression(c=0.001).fit(np.zeros((10, 2)), labels)
    assert model.intercept_ == pytest.approx(np.log(1 / 9))
    assert model.predict_proba([[0.0, 0.0]])[0, 1] == pytest.approx(0.1)


def test_subsets_are_nested_and_stratified():
    labels = np.r_[np.zeros(900), np.ones(100)]
    previous = set()
    for fraction, subset in nested_subsets(labels, seed=7):
        assert len(subset) == round(fraction * len(labels))
        assert labels[subset].mean() == pytest.approx(0.1)
        assert previous.issubset(subset)
        previous = set(subset)


def test_perturbations_are_paired_and_use_training_quantiles():
    fields = list(CONTINUOUS)
    train = pd.DataFrame(np.tile(np.arange(100.0)[:, None], (1, len(fields))), columns=fields)
    clean = train.assign(category="keep")
    original = clean.copy()
    result = dict(perturbed_inputs(train, clean, seed=42))
    np.testing.assert_allclose(
        result["noise_0.50"][fields] - clean[fields],
        5 * (result["noise_0.10"][fields] - clean[fields]),
        atol=1e-12,
    )
    q3 = 74.25
    iqr = 49.5
    changed = set()
    for key, expected in [("outliers_0.01", 1), ("outliers_0.05", 5), ("outliers_0.10", 10)]:
        altered = result[key]
        differences = altered[fields] != clean[fields]
        selected = set(altered.index[differences.any(axis=1)])
        assert len(selected) == differences.to_numpy().sum() == expected
        assert changed <= selected
        assert (altered[fields].to_numpy()[differences.to_numpy()] == q3 + 10 * iqr).all()
        changed = selected
    assert all(altered["category"].equals(clean["category"]) for altered in result.values())
    pd.testing.assert_frame_equal(clean, original)
