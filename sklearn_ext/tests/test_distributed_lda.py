"""Tests for merge_lda_models (distributed LDA merge)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn_ext.distributed_lda import merge_lda_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n_samples=300, n_features=6, n_classes=3, random_state=42):
    """Generate a reproducible classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_features, max(n_classes, 2)),
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    return X, y


def _partial_fit_on_chunks(X, y, n_chunks, classes=None, **lda_kwargs):
    """Return a list of independently partial_fit'd LDA models, one per chunk."""
    if classes is None:
        classes = np.unique(y)
    chunks_X = np.array_split(X, n_chunks)
    chunks_y = np.array_split(y, n_chunks)
    models = []
    for cx, cy in zip(chunks_X, chunks_y):
        m = LinearDiscriminantAnalysis(solver="svd", **lda_kwargs)
        m.partial_fit(cx, cy, classes=classes)
        models.append(m)
    return models


def _sequential_partial_fit(X, y, n_chunks, classes=None, **lda_kwargs):
    """Sequentially partial_fit one model on all chunks."""
    if classes is None:
        classes = np.unique(y)
    chunks_X = np.array_split(X, n_chunks)
    chunks_y = np.array_split(y, n_chunks)
    m = LinearDiscriminantAnalysis(solver="svd", **lda_kwargs)
    for cx, cy in zip(chunks_X, chunks_y):
        m.partial_fit(cx, cy, classes=classes)
    return m


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_chunks", [2, 4, 8])
def test_merge_equals_sequential_partial_fit(n_chunks):
    X, y = _make_data(n_samples=400, n_features=6, n_classes=3)
    classes = np.unique(y)

    models = _partial_fit_on_chunks(X, y, n_chunks, classes=classes)
    merged = merge_lda_models(models)

    seq = _sequential_partial_fit(X, y, n_chunks, classes=classes)

    assert_array_equal(merged.classes_, seq.classes_)
    assert_allclose(merged.means_, seq.means_, atol=1e-10)
    assert_allclose(np.asarray(merged._class_counts), np.asarray(seq._class_counts))
    assert_allclose(merged.priors_, seq.priors_, atol=1e-10)
    assert_allclose(merged.coef_, seq.coef_, atol=1e-8)
    assert_allclose(merged.intercept_, seq.intercept_, atol=1e-8)
    assert_array_equal(merged.predict(X), seq.predict(X))


def test_merge_equals_batch_fit():
    X, y = _make_data(n_samples=300, n_features=6, n_classes=3)
    classes = np.unique(y)

    models = _partial_fit_on_chunks(X, y, 3, classes=classes)
    merged = merge_lda_models(models)

    batch = LinearDiscriminantAnalysis(solver="svd").fit(X, y)

    # Merged predictions should match batch fit
    assert_array_equal(merged.predict(X), batch.predict(X))
    assert_allclose(merged.predict_proba(X), batch.predict_proba(X), atol=1e-6)


def test_merge_associativity():
    X, y = _make_data(n_samples=300, n_features=6, n_classes=3)
    classes = np.unique(y)
    models = _partial_fit_on_chunks(X, y, 3, classes=classes)
    A, B, C = models

    # merge(merge(A,B), C) vs merge(A, B, C)
    merged_ab = merge_lda_models([A, B])
    merged_abc_1 = merge_lda_models([merged_ab, C])
    merged_abc_2 = merge_lda_models([A, B, C])

    assert_allclose(merged_abc_1.coef_, merged_abc_2.coef_, atol=1e-8)
    assert_allclose(merged_abc_1.intercept_, merged_abc_2.intercept_, atol=1e-8)
    assert_array_equal(
        merged_abc_1.predict(X), merged_abc_2.predict(X)
    )


def test_merge_commutativity():
    X, y = _make_data(n_samples=300, n_features=6, n_classes=3)
    classes = np.unique(y)
    models = _partial_fit_on_chunks(X, y, 2, classes=classes)
    A, B = models

    m1 = merge_lda_models([A, B])
    m2 = merge_lda_models([B, A])

    assert_allclose(m1.coef_, m2.coef_, atol=1e-8)
    assert_allclose(m1.intercept_, m2.intercept_, atol=1e-8)
    assert_array_equal(m1.predict(X), m2.predict(X))


# ---------------------------------------------------------------------------
# Partial class coverage
# ---------------------------------------------------------------------------


def test_merge_disjoint_classes():
    """Workers see non-overlapping class subsets."""
    X, y = _make_data(n_samples=300, n_features=6, n_classes=4)
    all_classes = np.unique(y)

    # Worker A sees classes 0, 1; Worker B sees classes 2, 3
    mask_a = np.isin(y, all_classes[:2])
    mask_b = np.isin(y, all_classes[2:])

    m_a = LinearDiscriminantAnalysis(solver="svd")
    m_a.partial_fit(X[mask_a], y[mask_a], classes=all_classes)
    m_b = LinearDiscriminantAnalysis(solver="svd")
    m_b.partial_fit(X[mask_b], y[mask_b], classes=all_classes)

    merged = merge_lda_models([m_a, m_b])

    seq = LinearDiscriminantAnalysis(solver="svd")
    seq.partial_fit(X[mask_a], y[mask_a], classes=all_classes)
    seq.partial_fit(X[mask_b], y[mask_b], classes=all_classes)

    assert_array_equal(merged.classes_, seq.classes_)
    assert_allclose(merged.means_, seq.means_, atol=1e-10)
    assert_allclose(merged.coef_, seq.coef_, atol=1e-8)
    assert_array_equal(merged.predict(X), seq.predict(X))


def test_merge_partial_overlap():
    """Worker A sees {0,1,2}, Worker B sees {1,2,3}."""
    X, y = _make_data(n_samples=400, n_features=6, n_classes=4)
    all_classes = np.unique(y)

    mask_a = np.isin(y, all_classes[:3])
    mask_b = np.isin(y, all_classes[1:])

    m_a = LinearDiscriminantAnalysis(solver="svd")
    m_a.partial_fit(X[mask_a], y[mask_a], classes=all_classes)
    m_b = LinearDiscriminantAnalysis(solver="svd")
    m_b.partial_fit(X[mask_b], y[mask_b], classes=all_classes)

    merged = merge_lda_models([m_a, m_b])

    seq = LinearDiscriminantAnalysis(solver="svd")
    seq.partial_fit(X[mask_a], y[mask_a], classes=all_classes)
    seq.partial_fit(X[mask_b], y[mask_b], classes=all_classes)

    assert_allclose(merged.means_, seq.means_, atol=1e-10)
    assert_allclose(merged.coef_, seq.coef_, atol=1e-8)
    assert_array_equal(merged.predict(X), seq.predict(X))


def test_merge_single_class_per_worker():
    """Each worker sees exactly one class."""
    X, y = _make_data(n_samples=300, n_features=6, n_classes=3)
    all_classes = np.unique(y)

    models = []
    for c in all_classes:
        mask = y == c
        m = LinearDiscriminantAnalysis(solver="svd")
        m.partial_fit(X[mask], y[mask], classes=all_classes)
        models.append(m)

    merged = merge_lda_models(models)

    seq = LinearDiscriminantAnalysis(solver="svd")
    for c in all_classes:
        mask = y == c
        seq.partial_fit(X[mask], y[mask], classes=all_classes)

    assert_allclose(merged.means_, seq.means_, atol=1e-10)
    assert_allclose(merged.coef_, seq.coef_, atol=1e-8)
    assert_array_equal(merged.predict(X), seq.predict(X))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_merge_binary():
    """2 classes → verify 1D coef_/intercept_ reduction."""
    X, y = _make_data(n_samples=200, n_features=6, n_classes=2)
    classes = np.unique(y)

    models = _partial_fit_on_chunks(X, y, 2, classes=classes)
    merged = merge_lda_models(models)

    batch = LinearDiscriminantAnalysis(solver="svd").fit(X, y)

    assert merged.coef_.shape == (1, 6)
    assert merged.intercept_.shape == (1,)
    assert_array_equal(merged.predict(X), batch.predict(X))


def test_merge_single_sample_per_worker():
    """Numerical stability with minimal data per worker."""
    rng = np.random.RandomState(42)
    # Need at least n_classes + 1 samples total so N_total > n_classes
    n_classes = 3
    n_features = 4
    n_samples = n_classes + 2  # 5 samples total

    X = rng.randn(n_samples, n_features)
    y = np.array([0, 1, 2, 0, 1])
    classes = np.arange(n_classes)

    models = []
    for i in range(n_samples):
        m = LinearDiscriminantAnalysis(solver="svd")
        m.partial_fit(X[i:i+1], y[i:i+1], classes=classes)
        models.append(m)

    merged = merge_lda_models(models)

    seq = LinearDiscriminantAnalysis(solver="svd")
    for i in range(n_samples):
        seq.partial_fit(X[i:i+1], y[i:i+1], classes=classes)

    assert_allclose(merged.means_, seq.means_, atol=1e-10)
    assert_array_equal(merged.predict(X), seq.predict(X))


def test_merge_high_d():
    """D >> N: SVD rank truncation."""
    rng = np.random.RandomState(42)
    n_samples = 30
    n_features = 200
    n_classes = 3

    X = rng.randn(n_samples, n_features)
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    classes = np.arange(n_classes)

    models = _partial_fit_on_chunks(X, y, 3, classes=classes)
    merged = merge_lda_models(models)

    seq = _sequential_partial_fit(X, y, 3, classes=classes)

    assert_allclose(merged.means_, seq.means_, atol=1e-10)
    assert_array_equal(merged.predict(X), seq.predict(X))


def test_merge_transform():
    """transform() matches batch."""
    X, y = _make_data(n_samples=300, n_features=6, n_classes=3)
    classes = np.unique(y)

    models = _partial_fit_on_chunks(X, y, 3, classes=classes)
    merged = merge_lda_models(models)

    batch = LinearDiscriminantAnalysis(solver="svd").fit(X, y)

    assert_allclose(merged.transform(X), batch.transform(X), atol=1e-6)


def test_merge_predict_proba():
    """predict_proba() matches batch."""
    X, y = _make_data(n_samples=300, n_features=6, n_classes=3)
    classes = np.unique(y)

    models = _partial_fit_on_chunks(X, y, 3, classes=classes)
    merged = merge_lda_models(models)

    batch = LinearDiscriminantAnalysis(solver="svd").fit(X, y)

    assert_allclose(
        merged.predict_proba(X), batch.predict_proba(X), atol=1e-6
    )


def test_merge_store_covariance():
    """store_covariance=True works after merge."""
    X, y = _make_data(n_samples=300, n_features=6, n_classes=3)
    classes = np.unique(y)

    models = _partial_fit_on_chunks(
        X, y, 3, classes=classes, store_covariance=True
    )
    merged = merge_lda_models(models)

    batch = LinearDiscriminantAnalysis(
        solver="svd", store_covariance=True
    ).fit(X, y)

    assert hasattr(merged, "covariance_")
    assert_allclose(merged.covariance_, batch.covariance_, atol=1e-6)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_merge_raises_fewer_than_two():
    X, y = _make_data(n_samples=50, n_features=4, n_classes=2)
    classes = np.unique(y)
    m = LinearDiscriminantAnalysis(solver="svd")
    m.partial_fit(X, y, classes=classes)

    with pytest.raises(ValueError, match="at least 2"):
        merge_lda_models([m])


def test_merge_raises_incompatible_solver():
    X, y = _make_data(n_samples=100, n_features=4, n_classes=2)
    classes = np.unique(y)

    m1 = LinearDiscriminantAnalysis(solver="svd")
    m1.partial_fit(X, y, classes=classes)
    m2 = LinearDiscriminantAnalysis(solver="lsqr")
    m2.fit(X, y)

    with pytest.raises(ValueError, match="solver='lsqr'"):
        merge_lda_models([m1, m2])


def test_merge_raises_incompatible_features():
    rng = np.random.RandomState(42)
    classes = np.array([0, 1])
    X1 = rng.randn(50, 4)
    y1 = np.repeat([0, 1], 25)
    X2 = rng.randn(50, 6)
    y2 = np.repeat([0, 1], 25)

    m1 = LinearDiscriminantAnalysis(solver="svd")
    m1.partial_fit(X1, y1, classes=classes)
    m2 = LinearDiscriminantAnalysis(solver="svd")
    m2.partial_fit(X2, y2, classes=classes)

    with pytest.raises(ValueError, match="n_features_in_"):
        merge_lda_models([m1, m2])


def test_merge_raises_unfitted():
    """Model from fit() doesn't have streaming state."""
    X, y = _make_data(n_samples=100, n_features=4, n_classes=2)
    classes = np.unique(y)

    m1 = LinearDiscriminantAnalysis(solver="svd")
    m1.partial_fit(X, y, classes=classes)
    m2 = LinearDiscriminantAnalysis(solver="svd")
    m2.fit(X, y)

    with pytest.raises(ValueError, match="partial_fit"):
        merge_lda_models([m1, m2])


def test_merge_raises_incompatible_hyperparams():
    X, y = _make_data(n_samples=100, n_features=4, n_classes=2)
    classes = np.unique(y)

    m1 = LinearDiscriminantAnalysis(solver="svd", tol=1e-4)
    m1.partial_fit(X, y, classes=classes)
    m2 = LinearDiscriminantAnalysis(solver="svd", tol=1e-2)
    m2.partial_fit(X, y, classes=classes)

    with pytest.raises(ValueError, match="tol"):
        merge_lda_models([m1, m2])
