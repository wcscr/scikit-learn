import warnings

import numpy as np
import pytest
from scipy import linalg

from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, ShrunkCovariance, ledoit_wolf
from sklearn.datasets import make_blobs, make_classification
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    _cov,
)
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

# Data is just 6 separable points in the plane
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype="f")
y = np.array([1, 1, 1, 2, 2, 2])
y3 = np.array([1, 1, 2, 2, 3, 3])

# Degenerate data with only one feature (still should be separable)
X1 = np.array(
    [[-2], [-1], [-1], [1], [1], [2]],
    dtype="f",
)

# Data is just 9 separable points in the plane
X6 = np.array(
    [[0, 0], [-2, -2], [-2, -1], [-1, -1], [-1, -2], [1, 3], [1, 2], [2, 1], [2, 2]]
)
y6 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
y7 = np.array([1, 2, 3, 2, 3, 1, 2, 3, 1])

# Degenerate data with 1 feature (still should be separable)
X7 = np.array([[-3], [-2], [-1], [-1], [0], [1], [1], [2], [3]])

# Data that has zero variance in one dimension and needs regularization
X2 = np.array(
    [[-3, 0], [-2, 0], [-1, 0], [-1, 0], [0, 0], [1, 0], [1, 0], [2, 0], [3, 0]]
)

# One element class
y4 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2])

solver_shrinkage = [
    ("svd", None),
    ("lsqr", None),
    ("eigen", None),
    ("lsqr", "auto"),
    ("lsqr", 0),
    ("lsqr", 0.43),
    ("eigen", "auto"),
    ("eigen", 0),
    ("eigen", 0.43),
]


def test_lda_predict():
    # Test LDA classification.
    # This checks that LDA implements fit and predict and returns correct
    # values for simple toy data.
    for test_case in solver_shrinkage:
        solver, shrinkage = test_case
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        y_pred = clf.fit(X, y).predict(X)
        assert_array_equal(y_pred, y, "solver %s" % solver)

        # Assert that it works with 1D data
        y_pred1 = clf.fit(X1, y).predict(X1)
        assert_array_equal(y_pred1, y, "solver %s" % solver)

        # Test probability estimates
        y_proba_pred1 = clf.predict_proba(X1)
        assert_array_equal((y_proba_pred1[:, 1] > 0.5) + 1, y, "solver %s" % solver)
        y_log_proba_pred1 = clf.predict_log_proba(X1)
        assert_allclose(
            np.exp(y_log_proba_pred1),
            y_proba_pred1,
            rtol=1e-6,
            atol=1e-6,
            err_msg="solver %s" % solver,
        )

        # Primarily test for commit 2f34950 -- "reuse" of priors
        y_pred3 = clf.fit(X, y3).predict(X)
        # LDA shouldn't be able to separate those
        assert np.any(y_pred3 != y3), "solver %s" % solver

    clf = LinearDiscriminantAnalysis(solver="svd", shrinkage="auto")
    with pytest.raises(NotImplementedError):
        clf.fit(X, y)

    clf = LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage=0.1, covariance_estimator=ShrunkCovariance()
    )
    with pytest.raises(
        ValueError,
        match=(
            "covariance_estimator and shrinkage "
            "parameters are not None. "
            "Only one of the two can be set."
        ),
    ):
        clf.fit(X, y)

    # test bad solver with covariance_estimator
    clf = LinearDiscriminantAnalysis(solver="svd", covariance_estimator=LedoitWolf())
    with pytest.raises(
        ValueError, match="covariance estimator is not supported with svd"
    ):
        clf.fit(X, y)

    # test bad covariance estimator
    clf = LinearDiscriminantAnalysis(
        solver="lsqr", covariance_estimator=KMeans(n_clusters=2, n_init="auto")
    )
    with pytest.raises(ValueError):
        clf.fit(X, y)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("solver", ["svd", "lsqr", "eigen"])
def test_lda_predict_proba(solver, n_classes):
    def generate_dataset(n_samples, centers, covariances, random_state=None):
        """Generate a multivariate normal data given some centers and
        covariances"""
        rng = check_random_state(random_state)
        X = np.vstack(
            [
                rng.multivariate_normal(mean, cov, size=n_samples // len(centers))
                for mean, cov in zip(centers, covariances)
            ]
        )
        y = np.hstack(
            [[clazz] * (n_samples // len(centers)) for clazz in range(len(centers))]
        )
        return X, y

    blob_centers = np.array([[0, 0], [-10, 40], [-30, 30]])[:n_classes]
    blob_stds = np.array([[[10, 10], [10, 100]]] * len(blob_centers))
    X, y = generate_dataset(
        n_samples=90000, centers=blob_centers, covariances=blob_stds, random_state=42
    )
    lda = LinearDiscriminantAnalysis(
        solver=solver, store_covariance=True, shrinkage=None
    ).fit(X, y)
    # check that the empirical means and covariances are close enough to the
    # one used to generate the data
    assert_allclose(lda.means_, blob_centers, atol=1e-1)
    assert_allclose(lda.covariance_, blob_stds[0], atol=1)

    # implement the method to compute the probability given in The Elements
    # of Statistical Learning (cf. p.127, Sect. 4.4.5 "Logistic Regression
    # or LDA?")
    precision = linalg.inv(blob_stds[0])
    alpha_k = []
    alpha_k_0 = []
    for clazz in range(len(blob_centers) - 1):
        alpha_k.append(
            np.dot(precision, (blob_centers[clazz] - blob_centers[-1])[:, np.newaxis])
        )
        alpha_k_0.append(
            np.dot(
                -0.5 * (blob_centers[clazz] + blob_centers[-1])[np.newaxis, :],
                alpha_k[-1],
            )
        )

    sample = np.array([[-22, 22]])

    def discriminant_func(sample, coef, intercept, clazz):
        return np.exp(intercept[clazz] + np.dot(sample, coef[clazz])).item()

    prob = np.array(
        [
            float(
                discriminant_func(sample, alpha_k, alpha_k_0, clazz)
                / (
                    1
                    + sum(
                        [
                            discriminant_func(sample, alpha_k, alpha_k_0, clazz)
                            for clazz in range(n_classes - 1)
                        ]
                    )
                )
            )
            for clazz in range(n_classes - 1)
        ]
    )

    prob_ref = 1 - np.sum(prob)

    # check the consistency of the computed probability
    # all probabilities should sum to one
    prob_ref_2 = float(
        1
        / (
            1
            + sum(
                [
                    discriminant_func(sample, alpha_k, alpha_k_0, clazz)
                    for clazz in range(n_classes - 1)
                ]
            )
        )
    )

    assert prob_ref == pytest.approx(prob_ref_2)
    # check that the probability of LDA are close to the theoretical
    # probabilities
    assert_allclose(
        lda.predict_proba(sample), np.hstack([prob, prob_ref])[np.newaxis], atol=1e-2
    )


def test_lda_priors():
    # Test priors (negative priors)
    priors = np.array([0.5, -0.5])
    clf = LinearDiscriminantAnalysis(priors=priors)
    msg = "priors must be non-negative"

    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)

    # Test that priors passed as a list are correctly handled (run to see if
    # failure)
    clf = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
    clf.fit(X, y)

    # Test that priors always sum to 1
    priors = np.array([0.5, 0.6])
    prior_norm = np.array([0.45, 0.55])
    clf = LinearDiscriminantAnalysis(priors=priors)

    with pytest.warns(UserWarning):
        clf.fit(X, y)

    assert_array_almost_equal(clf.priors_, prior_norm, 2)


def test_lda_coefs():
    # Test if the coefficients of the solvers are approximately the same.
    n_features = 2
    n_classes = 2
    n_samples = 1000
    X, y = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=n_classes, random_state=11
    )

    clf_lda_svd = LinearDiscriminantAnalysis(solver="svd")
    clf_lda_lsqr = LinearDiscriminantAnalysis(solver="lsqr")
    clf_lda_eigen = LinearDiscriminantAnalysis(solver="eigen")

    clf_lda_svd.fit(X, y)
    clf_lda_lsqr.fit(X, y)
    clf_lda_eigen.fit(X, y)

    assert_array_almost_equal(clf_lda_svd.coef_, clf_lda_lsqr.coef_, 1)
    assert_array_almost_equal(clf_lda_svd.coef_, clf_lda_eigen.coef_, 1)
    assert_array_almost_equal(clf_lda_eigen.coef_, clf_lda_lsqr.coef_, 1)


def test_lda_transform():
    # Test LDA transform.
    clf = LinearDiscriminantAnalysis(solver="svd", n_components=1)
    X_transformed = clf.fit(X, y).transform(X)
    assert X_transformed.shape[1] == 1
    clf = LinearDiscriminantAnalysis(solver="eigen", n_components=1)
    X_transformed = clf.fit(X, y).transform(X)
    assert X_transformed.shape[1] == 1

    clf = LinearDiscriminantAnalysis(solver="lsqr", n_components=1)
    clf.fit(X, y)
    msg = "transform not implemented for 'lsqr'"

    with pytest.raises(NotImplementedError, match=msg):
        clf.transform(X)


def test_lda_explained_variance_ratio():
    # Test if the sum of the normalized eigen vectors values equals 1,
    # Also tests whether the explained_variance_ratio_ formed by the
    # eigen solver is the same as the explained_variance_ratio_ formed
    # by the svd solver

    state = np.random.RandomState(0)
    X = state.normal(loc=0, scale=100, size=(40, 20))
    y = state.randint(0, 3, size=(40,))

    clf_lda_eigen = LinearDiscriminantAnalysis(solver="eigen")
    clf_lda_eigen.fit(X, y)
    assert_almost_equal(clf_lda_eigen.explained_variance_ratio_.sum(), 1.0, 3)
    assert clf_lda_eigen.explained_variance_ratio_.shape == (2,), (
        "Unexpected length for explained_variance_ratio_"
    )

    clf_lda_svd = LinearDiscriminantAnalysis(solver="svd")
    clf_lda_svd.fit(X, y)
    assert_almost_equal(clf_lda_svd.explained_variance_ratio_.sum(), 1.0, 3)
    assert clf_lda_svd.explained_variance_ratio_.shape == (2,), (
        "Unexpected length for explained_variance_ratio_"
    )

    assert_array_almost_equal(
        clf_lda_svd.explained_variance_ratio_, clf_lda_eigen.explained_variance_ratio_
    )


def test_lda_orthogonality():
    # arrange four classes with their means in a kite-shaped pattern
    # the longer distance should be transformed to the first component, and
    # the shorter distance to the second component.
    means = np.array([[0, 0, -1], [0, 2, 0], [0, -2, 0], [0, 0, 5]])

    # We construct perfectly symmetric distributions, so the LDA can estimate
    # precise means.
    scatter = np.array(
        [
            [0.1, 0, 0],
            [-0.1, 0, 0],
            [0, 0.1, 0],
            [0, -0.1, 0],
            [0, 0, 0.1],
            [0, 0, -0.1],
        ]
    )

    X = (means[:, np.newaxis, :] + scatter[np.newaxis, :, :]).reshape((-1, 3))
    y = np.repeat(np.arange(means.shape[0]), scatter.shape[0])

    # Fit LDA and transform the means
    clf = LinearDiscriminantAnalysis(solver="svd").fit(X, y)
    means_transformed = clf.transform(means)

    d1 = means_transformed[3] - means_transformed[0]
    d2 = means_transformed[2] - means_transformed[1]
    d1 /= np.sqrt(np.sum(d1**2))
    d2 /= np.sqrt(np.sum(d2**2))

    # the transformed within-class covariance should be the identity matrix
    assert_almost_equal(np.cov(clf.transform(scatter).T), np.eye(2))

    # the means of classes 0 and 3 should lie on the first component
    assert_almost_equal(np.abs(np.dot(d1[:2], [1, 0])), 1.0)

    # the means of classes 1 and 2 should lie on the second component
    assert_almost_equal(np.abs(np.dot(d2[:2], [0, 1])), 1.0)


def test_lda_scaling():
    # Test if classification works correctly with differently scaled features.
    n = 100
    rng = np.random.RandomState(1234)
    # use uniform distribution of features to make sure there is absolutely no
    # overlap between classes.
    x1 = rng.uniform(-1, 1, (n, 3)) + [-10, 0, 0]
    x2 = rng.uniform(-1, 1, (n, 3)) + [10, 0, 0]
    x = np.vstack((x1, x2)) * [1, 100, 10000]
    y = [-1] * n + [1] * n

    for solver in ("svd", "lsqr", "eigen"):
        clf = LinearDiscriminantAnalysis(solver=solver)
        # should be able to separate the data perfectly
        assert clf.fit(x, y).score(x, y) == 1.0, "using covariance: %s" % solver


def test_lda_store_covariance():
    # Test for solver 'lsqr' and 'eigen'
    # 'store_covariance' has no effect on 'lsqr' and 'eigen' solvers
    for solver in ("lsqr", "eigen"):
        clf = LinearDiscriminantAnalysis(solver=solver).fit(X6, y6)
        assert hasattr(clf, "covariance_")

        # Test the actual attribute:
        clf = LinearDiscriminantAnalysis(solver=solver, store_covariance=True).fit(
            X6, y6
        )
        assert hasattr(clf, "covariance_")

        assert_array_almost_equal(
            clf.covariance_, np.array([[0.422222, 0.088889], [0.088889, 0.533333]])
        )

    # Test for SVD solver, the default is to not set the covariances_ attribute
    clf = LinearDiscriminantAnalysis(solver="svd").fit(X6, y6)
    assert not hasattr(clf, "covariance_")

    # Test the actual attribute:
    clf = LinearDiscriminantAnalysis(solver=solver, store_covariance=True).fit(X6, y6)
    assert hasattr(clf, "covariance_")

    assert_array_almost_equal(
        clf.covariance_, np.array([[0.422222, 0.088889], [0.088889, 0.533333]])
    )


@pytest.mark.parametrize("seed", range(10))
def test_lda_shrinkage(seed):
    # Test that shrunk covariance estimator and shrinkage parameter behave the
    # same
    rng = np.random.RandomState(seed)
    X = rng.rand(100, 10)
    y = rng.randint(3, size=(100))
    c1 = LinearDiscriminantAnalysis(store_covariance=True, shrinkage=0.5, solver="lsqr")
    c2 = LinearDiscriminantAnalysis(
        store_covariance=True,
        covariance_estimator=ShrunkCovariance(shrinkage=0.5),
        solver="lsqr",
    )
    c1.fit(X, y)
    c2.fit(X, y)
    assert_allclose(c1.means_, c2.means_)
    assert_allclose(c1.covariance_, c2.covariance_)


def test_lda_ledoitwolf():
    # When shrinkage="auto" current implementation uses ledoitwolf estimation
    # of covariance after standardizing the data. This checks that it is indeed
    # the case
    class StandardizedLedoitWolf:
        def fit(self, X):
            sc = StandardScaler()  # standardize features
            X_sc = sc.fit_transform(X)
            s = ledoit_wolf(X_sc)[0]
            # rescale
            s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            self.covariance_ = s

    rng = np.random.RandomState(0)
    X = rng.rand(100, 10)
    y = rng.randint(3, size=(100,))
    c1 = LinearDiscriminantAnalysis(
        store_covariance=True, shrinkage="auto", solver="lsqr"
    )
    c2 = LinearDiscriminantAnalysis(
        store_covariance=True,
        covariance_estimator=StandardizedLedoitWolf(),
        solver="lsqr",
    )
    c1.fit(X, y)
    c2.fit(X, y)
    assert_allclose(c1.means_, c2.means_)
    assert_allclose(c1.covariance_, c2.covariance_)


@pytest.mark.parametrize("n_features", [3, 5])
@pytest.mark.parametrize("n_classes", [5, 3])
def test_lda_dimension_warning(n_classes, n_features):
    rng = check_random_state(0)
    n_samples = 10
    X = rng.randn(n_samples, n_features)
    # we create n_classes labels by repeating and truncating a
    # range(n_classes) until n_samples
    y = np.tile(range(n_classes), n_samples // n_classes + 1)[:n_samples]
    max_components = min(n_features, n_classes - 1)

    for n_components in [max_components - 1, None, max_components]:
        # if n_components <= min(n_classes - 1, n_features), no warning
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(X, y)

    for n_components in [max_components + 1, max(n_features, n_classes - 1) + 1]:
        # if n_components > min(n_classes - 1, n_features), raise error.
        # We test one unit higher than max_components, and then something
        # larger than both n_features and n_classes - 1 to ensure the test
        # works for any value of n_component
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        msg = "n_components cannot be larger than "
        with pytest.raises(ValueError, match=msg):
            lda.fit(X, y)


@pytest.mark.parametrize(
    "data_type, expected_type",
    [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ],
)
def test_lda_dtype_match(data_type, expected_type):
    for solver, shrinkage in solver_shrinkage:
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        clf.fit(X.astype(data_type), y.astype(data_type))
        assert clf.coef_.dtype == expected_type


def test_lda_numeric_consistency_float32_float64():
    for solver, shrinkage in solver_shrinkage:
        clf_32 = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        clf_32.fit(X.astype(np.float32), y.astype(np.float32))
        clf_64 = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        clf_64.fit(X.astype(np.float64), y.astype(np.float64))

        # Check value consistency between types
        rtol = 1e-6
        assert_allclose(clf_32.coef_, clf_64.coef_, rtol=rtol)


@pytest.mark.parametrize("solver", ["svd", "eigen"])
def test_qda(solver):
    # QDA classification.
    # This checks that QDA implements fit and predict and returns
    # correct values for a simple toy dataset.
    clf = QuadraticDiscriminantAnalysis(solver=solver)
    y_pred = clf.fit(X6, y6).predict(X6)
    assert_array_equal(y_pred, y6)

    # Assure that it works with 1D data
    y_pred1 = clf.fit(X7, y6).predict(X7)
    assert_array_equal(y_pred1, y6)

    # Test probas estimates
    y_proba_pred1 = clf.predict_proba(X7)
    assert_array_equal((y_proba_pred1[:, 1] > 0.5) + 1, y6)
    y_log_proba_pred1 = clf.predict_log_proba(X7)
    assert_array_almost_equal(np.exp(y_log_proba_pred1), y_proba_pred1, 8)

    y_pred3 = clf.fit(X6, y7).predict(X6)
    # QDA shouldn't be able to separate those
    assert np.any(y_pred3 != y7)

    # Classes should have at least 2 elements
    with pytest.raises(ValueError):
        clf.fit(X6, y4)


def test_qda_covariance_estimator():
    # Test that the correct errors are raised when using inappropriate
    # covariance estimators or shrinkage parameters with QDA.
    clf = QuadraticDiscriminantAnalysis(solver="svd", shrinkage="auto")
    with pytest.raises(NotImplementedError):
        clf.fit(X, y)

    clf = QuadraticDiscriminantAnalysis(
        solver="eigen", shrinkage=0.1, covariance_estimator=ShrunkCovariance()
    )
    with pytest.raises(
        ValueError,
        match=(
            "covariance_estimator and shrinkage parameters are not None. "
            "Only one of the two can be set."
        ),
    ):
        clf.fit(X, y)

    # test bad solver with covariance_estimator
    clf = QuadraticDiscriminantAnalysis(solver="svd", covariance_estimator=LedoitWolf())
    with pytest.raises(
        ValueError, match="covariance_estimator is not supported with solver='svd'"
    ):
        clf.fit(X, y)

    # test bad covariance estimator
    clf = QuadraticDiscriminantAnalysis(
        solver="eigen", covariance_estimator=KMeans(n_clusters=2, n_init="auto")
    )
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_qda_ledoitwolf(global_random_seed):
    # When shrinkage="auto" current implementation uses ledoitwolf estimation
    # of covariance after standardizing the data. This checks that it is indeed
    # the case
    class StandardizedLedoitWolf:
        def fit(self, X):
            sc = StandardScaler()  # standardize features
            X_sc = sc.fit_transform(X)
            s = ledoit_wolf(X_sc)[0]
            # rescale
            s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            self.covariance_ = s

    rng = np.random.RandomState(global_random_seed)
    X = rng.rand(100, 10)
    y = rng.randint(3, size=(100,))
    c1 = QuadraticDiscriminantAnalysis(
        store_covariance=True, shrinkage="auto", solver="eigen"
    )
    c2 = QuadraticDiscriminantAnalysis(
        store_covariance=True,
        covariance_estimator=StandardizedLedoitWolf(),
        solver="eigen",
    )
    c1.fit(X, y)
    c2.fit(X, y)
    assert_allclose(c1.means_, c2.means_)
    assert_allclose(c1.covariance_, c2.covariance_)


def test_qda_coefs(global_random_seed):
    # Test if the coefficients of the solvers are approximately the same.
    n_features = 2
    n_classes = 2
    n_samples = 3000
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=[1.0, 3.0],
        random_state=global_random_seed,
    )

    clf_svd = QuadraticDiscriminantAnalysis(solver="svd")
    clf_eigen = QuadraticDiscriminantAnalysis(solver="eigen")

    clf_svd.fit(X, y)
    clf_eigen.fit(X, y)

    for class_idx in range(n_classes):
        assert_allclose(
            np.abs(clf_svd.rotations_[class_idx]),
            np.abs(clf_eigen.rotations_[class_idx]),
            rtol=1e-3,
            err_msg=f"SVD and Eigen rotations differ for class {class_idx}",
        )
        assert_allclose(
            clf_svd.scalings_[class_idx],
            clf_eigen.scalings_[class_idx],
            rtol=1e-3,
            err_msg=f"SVD and Eigen scalings differ for class {class_idx}",
        )


def test_qda_priors():
    clf = QuadraticDiscriminantAnalysis()
    y_pred = clf.fit(X6, y6).predict(X6)
    n_pos = np.sum(y_pred == 2)

    neg = 1e-10
    clf = QuadraticDiscriminantAnalysis(priors=np.array([neg, 1 - neg]))
    y_pred = clf.fit(X6, y6).predict(X6)
    n_pos2 = np.sum(y_pred == 2)

    assert n_pos2 > n_pos


@pytest.mark.parametrize("priors_type", ["list", "tuple", "array"])
def test_qda_prior_type(priors_type):
    """Check that priors accept array-like."""
    priors = [0.5, 0.5]
    clf = QuadraticDiscriminantAnalysis(
        priors=_convert_container([0.5, 0.5], priors_type)
    ).fit(X6, y6)
    assert isinstance(clf.priors_, np.ndarray)
    assert_array_equal(clf.priors_, priors)


def test_qda_prior_copy():
    """Check that altering `priors` without `fit` doesn't change `priors_`"""
    priors = np.array([0.5, 0.5])
    qda = QuadraticDiscriminantAnalysis(priors=priors).fit(X, y)

    # we expect the following
    assert_array_equal(qda.priors_, qda.priors)

    # altering `priors` without `fit` should not change `priors_`
    priors[0] = 0.2
    assert qda.priors_[0] != qda.priors[0]


def test_qda_store_covariance():
    # The default is to not set the covariances_ attribute
    clf = QuadraticDiscriminantAnalysis().fit(X6, y6)
    assert not hasattr(clf, "covariance_")

    # Test the actual attribute:
    clf = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X6, y6)
    assert hasattr(clf, "covariance_")

    assert_array_almost_equal(clf.covariance_[0], np.array([[0.7, 0.45], [0.45, 0.7]]))

    assert_array_almost_equal(
        clf.covariance_[1],
        np.array([[0.33333333, -0.33333333], [-0.33333333, 0.66666667]]),
    )


@pytest.mark.parametrize("solver", ["svd", "eigen"])
def test_qda_regularization(global_random_seed, solver):
    # The default is reg_param=0. and will cause issues when there is a
    # constant variable.
    rng = np.random.default_rng(global_random_seed)

    # Fitting on data with constant variable without regularization
    # triggers a LinAlgError.
    msg = r"The covariance matrix of class .+ is not full rank."
    clf = QuadraticDiscriminantAnalysis(solver=solver)
    with pytest.raises(linalg.LinAlgError, match=msg):
        clf.fit(X2, y6)

    with pytest.raises(AttributeError):
        y_pred = clf.predict(X2)

    # Adding a little regularization fixes the fit time error.
    if solver == "svd":
        clf = QuadraticDiscriminantAnalysis(solver=solver, reg_param=0.01)
    elif solver == "eigen":
        clf = QuadraticDiscriminantAnalysis(solver=solver, shrinkage=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
    clf.fit(X2, y6)
    y_pred = clf.predict(X2)
    assert_array_equal(y_pred, y6)

    # LinAlgError should also be there for the n_samples_in_a_class <
    # n_features case.
    X = rng.normal(size=(9, 4))
    y = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2])

    clf = QuadraticDiscriminantAnalysis(solver=solver)
    if solver == "svd":
        msg2 = msg + " When using `solver='svd'`"
    elif solver == "eigen":
        msg2 = msg

    with pytest.raises(linalg.LinAlgError, match=msg2):
        clf.fit(X, y)

    # The error will persist even with regularization for SVD
    # because the number of singular values is limited by n_samples_in_a_class.
    if solver == "svd":
        clf = QuadraticDiscriminantAnalysis(solver=solver, reg_param=0.3)
        with pytest.raises(linalg.LinAlgError, match=msg2):
            clf.fit(X, y)
    # The warning will be gone for Eigen with regularization, because
    # the covariance matrix will be full-rank.
    elif solver == "eigen":
        clf = QuadraticDiscriminantAnalysis(solver=solver, shrinkage=0.3)
        clf.fit(X, y)


def test_covariance():
    x, y = make_blobs(n_samples=100, n_features=5, centers=1, random_state=42)

    # make features correlated
    x = np.dot(x, np.arange(x.shape[1] ** 2).reshape(x.shape[1], x.shape[1]))

    c_e = _cov(x, "empirical")
    assert_almost_equal(c_e, c_e.T)

    c_s = _cov(x, "auto")
    assert_almost_equal(c_s, c_s.T)


@pytest.mark.parametrize("solver", ["svd", "lsqr", "eigen"])
def test_raises_value_error_on_same_number_of_classes_and_samples(solver):
    """
    Tests that if the number of samples equals the number
    of classes, a ValueError is raised.
    """
    X = np.array([[0.5, 0.6], [0.6, 0.5]])
    y = np.array(["a", "b"])
    clf = LinearDiscriminantAnalysis(solver=solver)
    with pytest.raises(ValueError, match="The number of samples must be more"):
        clf.fit(X, y)


@pytest.mark.parametrize("solver", ["svd", "eigen"])
def test_raises_value_error_on_one_sample_per_class(solver):
    """
    Tests that if a class has one sample, a ValueError is raised.
    """
    X = np.array([[0.5, 0.6], [0.6, 0.5], [0.4, 0.4], [0.6, 0.5]])
    y = np.array(["a", "a", "a", "b"])
    clf = QuadraticDiscriminantAnalysis(solver=solver)
    with pytest.raises(ValueError, match="y has only 1 sample in class"):
        clf.fit(X, y)


def test_get_feature_names_out():
    """Check get_feature_names_out uses class name as prefix."""

    est = LinearDiscriminantAnalysis().fit(X, y)
    names_out = est.get_feature_names_out()

    class_name_lower = "LinearDiscriminantAnalysis".lower()
    expected_names_out = np.array(
        [
            f"{class_name_lower}{i}"
            for i in range(est.explained_variance_ratio_.shape[0])
        ],
        dtype=object,
    )
    assert_array_equal(names_out, expected_names_out)


@pytest.mark.parametrize("n_features", [25])
@pytest.mark.parametrize("train_size", [100])
@pytest.mark.parametrize("solver_no_shrinkage", ["svd", "eigen"])
def test_qda_shrinkage_performance(
    global_random_seed, n_features, train_size, solver_no_shrinkage
):
    # Test that QDA with shrinkage performs better than without shrinkage on
    # a case where there's a small number of samples per class relative to
    # the number of features.
    n_samples = 1000
    n_features = n_features

    rng = np.random.default_rng(global_random_seed)

    # Sample from two Gaussians with different variances and same null means.
    vars1 = rng.uniform(2.0, 3.0, size=n_features)
    vars2 = rng.uniform(0.2, 1.0, size=n_features)

    X = np.concatenate(
        [
            np.random.randn(n_samples // 2, n_features) * np.sqrt(vars1),
            np.random.randn(n_samples // 2, n_features) * np.sqrt(vars2),
        ],
        axis=0,
    )
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Use small training sets to illustrate the regularization effect of
    # covariance shrinkage.
    cv = ShuffleSplit(n_splits=5, train_size=train_size, random_state=0)
    qda_shrinkage = QuadraticDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    qda_no_shrinkage = QuadraticDiscriminantAnalysis(
        solver=solver_no_shrinkage, shrinkage=None
    )

    scores_no_shrinkage = cross_val_score(
        qda_no_shrinkage, X, y, cv=cv, scoring="d2_brier_score"
    )
    scores_shrinkage = cross_val_score(
        qda_shrinkage, X, y, cv=cv, scoring="d2_brier_score"
    )

    assert scores_shrinkage.mean() > 0.9
    assert scores_no_shrinkage.mean() < 0.6


# ---------------------------------------------------------------------------
# Tests for LinearDiscriminantAnalysis.partial_fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver, shrinkage",
    [("eigen", None), ("eigen", 0.5), ("lsqr", None), ("lsqr", 0.5), ("svd", None)],
)
def test_lda_partial_fit_batch_equivalence(solver, shrinkage):
    """partial_fit over chunks must equal batch fit (exact online learning)."""
    X_full, y_full = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )

    clf_batch = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    clf_batch.fit(X_full, y_full)

    clf_online = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    classes = np.unique(y_full)
    chunk_size = 50
    for i in range(0, len(X_full), chunk_size):
        X_chunk = X_full[i : i + chunk_size]
        y_chunk = y_full[i : i + chunk_size]
        clf_online.partial_fit(X_chunk, y_chunk, classes=classes if i == 0 else None)

    assert_allclose(clf_online.means_, clf_batch.means_, atol=1e-7)
    assert_allclose(clf_online.priors_, clf_batch.priors_, atol=1e-7)
    assert_allclose(clf_online.coef_, clf_batch.coef_, atol=1e-5)
    assert_allclose(clf_online.intercept_, clf_batch.intercept_, atol=1e-5)
    if solver != "svd":
        assert_allclose(clf_online.covariance_, clf_batch.covariance_, atol=1e-7)


@pytest.mark.parametrize("solver", ["eigen", "lsqr", "svd"])
def test_lda_partial_fit_missing_classes_in_chunk(solver):
    """Chunks that omit a class must not corrupt historical statistics."""
    X_full, y_full = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=7,
    )

    clf_batch = LinearDiscriminantAnalysis(solver=solver)
    clf_batch.fit(X_full, y_full)

    clf_online = LinearDiscriminantAnalysis(solver=solver)
    classes = np.unique(y_full)

    # Feed data in per-class chunks so each chunk is missing 2 classes
    for c in classes:
        mask = y_full == c
        clf_online.partial_fit(
            X_full[mask],
            y_full[mask],
            classes=classes if c == classes[0] else None,
        )

    assert_allclose(clf_online.means_, clf_batch.means_, atol=1e-7)
    assert_allclose(clf_online.coef_, clf_batch.coef_, atol=1e-5)
    assert_allclose(clf_online.intercept_, clf_batch.intercept_, atol=1e-5)
    if solver != "svd":
        assert_allclose(clf_online.covariance_, clf_batch.covariance_, atol=1e-7)


@pytest.mark.parametrize("solver", ["eigen", "lsqr", "svd"])
def test_lda_partial_fit_single_sample_chunks(solver):
    """Single-sample chunks must not break covariance tracking."""
    X_full, y_full = make_classification(
        n_samples=100,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=0,
    )

    clf_batch = LinearDiscriminantAnalysis(solver=solver)
    clf_batch.fit(X_full, y_full)

    clf_online = LinearDiscriminantAnalysis(solver=solver)
    classes = np.unique(y_full)
    for i in range(len(X_full)):
        clf_online.partial_fit(
            X_full[i : i + 1], y_full[i : i + 1],
            classes=classes if i == 0 else None,
        )

    assert_allclose(clf_online.means_, clf_batch.means_, atol=1e-7)
    assert_allclose(clf_online.coef_, clf_batch.coef_, atol=1e-5)
    assert_allclose(clf_online.intercept_, clf_batch.intercept_, atol=1e-5)
    if solver != "svd":
        assert_allclose(clf_online.covariance_, clf_batch.covariance_, atol=1e-7)


def test_lda_partial_fit_collinear_features():
    """Collinear features must behave identically in batch and online modes.

    Only the 'lsqr' solver is tested here because the 'eigen' solver requires
    a positive definite within-class covariance matrix, which collinear
    features cannot provide.
    """
    rng = np.random.RandomState(42)
    n_samples = 200
    X_base = rng.randn(n_samples, 3)
    # Add a perfectly collinear 4th feature
    X_full = np.column_stack([X_base, X_base[:, 0] + X_base[:, 1]])
    y_full = (X_base[:, 0] > 0).astype(int)

    clf_batch = LinearDiscriminantAnalysis(solver="lsqr")
    clf_batch.fit(X_full, y_full)

    clf_online = LinearDiscriminantAnalysis(solver="lsqr")
    classes = np.unique(y_full)
    chunk_size = 40
    for i in range(0, n_samples, chunk_size):
        clf_online.partial_fit(
            X_full[i : i + chunk_size],
            y_full[i : i + chunk_size],
            classes=classes if i == 0 else None,
        )

    assert_allclose(clf_online.covariance_, clf_batch.covariance_, atol=1e-7)
    assert_allclose(clf_online.coef_, clf_batch.coef_, atol=1e-5)
    assert_allclose(clf_online.intercept_, clf_batch.intercept_, atol=1e-5)



def test_lda_partial_fit_raises_auto_shrinkage():
    """shrinkage='auto' must raise NotImplementedError."""
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    with pytest.raises(NotImplementedError, match="shrinkage='auto'"):
        clf.partial_fit(X, y, classes=np.unique(y))


def test_lda_partial_fit_raises_covariance_estimator():
    """Custom covariance_estimator must raise NotImplementedError."""
    clf = LinearDiscriminantAnalysis(
        solver="lsqr", covariance_estimator=ShrunkCovariance()
    )
    with pytest.raises(NotImplementedError, match="covariance_estimator"):
        clf.partial_fit(X, y, classes=np.unique(y))


def test_lda_partial_fit_raises_missing_classes():
    """First call without classes= must raise ValueError."""
    clf = LinearDiscriminantAnalysis(solver="lsqr")
    with pytest.raises(ValueError, match="classes must be passed on the first call"):
        clf.partial_fit(X, y)


@pytest.mark.parametrize("solver", ["eigen", "lsqr", "svd"])
def test_lda_partial_fit_binary(solver):
    """Binary classification must produce 1D coef_ / intercept_."""
    X_full, y_full = make_classification(
        n_samples=200, n_features=4, n_informative=4, n_redundant=0,
        n_classes=2, random_state=1,
    )

    clf_batch = LinearDiscriminantAnalysis(solver=solver)
    clf_batch.fit(X_full, y_full)

    clf_online = LinearDiscriminantAnalysis(solver=solver)
    classes = np.unique(y_full)
    clf_online.partial_fit(X_full, y_full, classes=classes)

    assert clf_online.coef_.shape == clf_batch.coef_.shape
    assert clf_online.intercept_.shape == clf_batch.intercept_.shape
    assert_allclose(clf_online.coef_, clf_batch.coef_, atol=1e-7)
    assert_allclose(clf_online.intercept_, clf_batch.intercept_, atol=1e-7)


@pytest.mark.parametrize("solver", ["eigen", "lsqr", "svd"])
def test_lda_partial_fit_predict(solver):
    """partial_fit model predictions must match batch fit predictions."""
    X_full, y_full = make_classification(
        n_samples=300,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )

    clf_batch = LinearDiscriminantAnalysis(solver=solver)
    clf_batch.fit(X_full, y_full)

    clf_online = LinearDiscriminantAnalysis(solver=solver)
    classes = np.unique(y_full)
    clf_online.partial_fit(X_full, y_full, classes=classes)

    assert_array_equal(clf_online.predict(X_full), clf_batch.predict(X_full))
    assert_allclose(
        clf_online.predict_proba(X_full),
        clf_batch.predict_proba(X_full),
        atol=1e-7,
    )


@pytest.mark.parametrize("solver", ["eigen", "lsqr", "svd"])
def test_lda_partial_fit_after_fit(solver):
    """partial_fit after fit must not crash and must work correctly."""
    X_full, y_full = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    classes = np.unique(y_full)

    # First fit, then partial_fit
    clf = LinearDiscriminantAnalysis(solver=solver)
    clf.fit(X_full, y_full)
    clf.partial_fit(X_full, y_full)

    # Compare against a fresh partial_fit-only estimator
    clf_fresh = LinearDiscriminantAnalysis(solver=solver)
    clf_fresh.partial_fit(X_full, y_full, classes=classes)

    assert_allclose(clf.means_, clf_fresh.means_, atol=1e-7)
    assert_allclose(clf.coef_, clf_fresh.coef_, atol=1e-7)
    assert_allclose(clf.intercept_, clf_fresh.intercept_, atol=1e-7)


def test_lda_partial_fit_raises_unknown_labels():
    """Unknown labels in y must raise ValueError."""
    clf = LinearDiscriminantAnalysis(solver="lsqr")
    clf.partial_fit(X, y, classes=np.array([1, 2]))

    X_new = np.array([[0.0, 0.0]])
    y_new = np.array([99])
    with pytest.raises(ValueError, match="do not exist in the initial"):
        clf.partial_fit(X_new, y_new)


@pytest.mark.parametrize("solver", ["eigen", "lsqr", "svd"])
def test_lda_partial_fit_honors_priors(solver):
    """User-supplied priors must be used instead of count-based priors."""
    X_full, y_full = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    classes = np.unique(y_full)
    explicit_priors = [0.3, 0.7]

    clf_priors = LinearDiscriminantAnalysis(
        solver=solver, priors=explicit_priors
    )
    clf_priors.partial_fit(X_full, y_full, classes=classes)

    clf_no_priors = LinearDiscriminantAnalysis(solver=solver)
    clf_no_priors.partial_fit(X_full, y_full, classes=classes)

    assert_allclose(clf_priors.priors_, explicit_priors, atol=1e-10)
    # intercept_ should differ when priors differ
    assert not np.allclose(
        clf_priors.intercept_,
        clf_no_priors.intercept_,
        atol=1e-5,
    )


@pytest.mark.parametrize("solver", ["lsqr", "svd"])
def test_lda_partial_fit_early_return_not_fitted(solver):
    """Early-return partial_fit must raise NotFittedError on predict."""
    from sklearn.exceptions import NotFittedError

    # Only one class seen => early return
    clf = LinearDiscriminantAnalysis(solver=solver)
    clf.partial_fit(X[:3], np.array([1, 1, 1]), classes=np.array([1, 2]))

    with pytest.raises(NotFittedError):
        clf.predict(X)


def test_lda_partial_fit_lsqr_low_samples():
    """lsqr must work when N_total - n_classes < n_features."""
    rng = np.random.RandomState(42)
    n_features = 10
    # 4 samples, 2 classes => within_df = 4 - 2 = 2 < 10
    X_small = rng.randn(4, n_features)
    y_small = np.array([0, 0, 1, 1])
    classes = np.array([0, 1])

    clf = LinearDiscriminantAnalysis(solver="lsqr")
    clf.partial_fit(X_small, y_small, classes=classes)

    # Should have coef_ and be usable for prediction
    assert hasattr(clf, "coef_")
    predictions = clf.predict(X_small)
    assert predictions.shape == (4,)


def test_lda_svd_partial_fit_transform_equivalence():
    """SVD partial_fit transform output must match batch fit."""
    X_full, y_full = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )

    clf_batch = LinearDiscriminantAnalysis(solver="svd")
    clf_batch.fit(X_full, y_full)

    clf_online = LinearDiscriminantAnalysis(solver="svd")
    classes = np.unique(y_full)
    chunk_size = 50
    for i in range(0, len(X_full), chunk_size):
        X_chunk = X_full[i : i + chunk_size]
        y_chunk = y_full[i : i + chunk_size]
        clf_online.partial_fit(X_chunk, y_chunk, classes=classes if i == 0 else None)

    X_batch = clf_batch.transform(X_full)
    X_online = clf_online.transform(X_full)

    # Correct for SVD sign ambiguity (columns may be sign-flipped)
    for col in range(X_batch.shape[1]):
        if np.dot(X_batch[:, col], X_online[:, col]) < 0:
            X_online[:, col] *= -1

    assert_allclose(X_online, X_batch, atol=1e-5)


def test_lda_svd_partial_fit_collinear():
    """SVD partial_fit handles collinear features via rank truncation."""
    rng = np.random.RandomState(42)
    n_samples = 200
    X_base = rng.randn(n_samples, 3)
    # Add a perfectly collinear 4th feature
    X_full = np.column_stack([X_base, X_base[:, 0] + X_base[:, 1]])
    y_full = (X_base[:, 0] > 0).astype(int)

    clf_batch = LinearDiscriminantAnalysis(solver="svd")
    clf_batch.fit(X_full, y_full)

    clf_online = LinearDiscriminantAnalysis(solver="svd")
    classes = np.unique(y_full)
    chunk_size = 40
    for i in range(0, n_samples, chunk_size):
        clf_online.partial_fit(
            X_full[i : i + chunk_size],
            y_full[i : i + chunk_size],
            classes=classes if i == 0 else None,
        )

    # Predictions should match
    assert_array_equal(clf_online.predict(X_full), clf_batch.predict(X_full))


def test_lda_svd_partial_fit_store_covariance():
    """SVD partial_fit with store_covariance=True produces correct covariance_."""
    X_full, y_full = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    # Batch reference with store_covariance
    clf_batch = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    clf_batch.fit(X_full, y_full)

    # Streaming with store_covariance
    clf_online = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    classes = np.unique(y_full)
    chunk_size = 50
    for i in range(0, len(X_full), chunk_size):
        clf_online.partial_fit(
            X_full[i : i + chunk_size],
            y_full[i : i + chunk_size],
            classes=classes if i == 0 else None,
        )

    assert hasattr(clf_online, "covariance_")
    assert_allclose(clf_online.covariance_, clf_batch.covariance_, atol=1e-10)

    # Without store_covariance, covariance_ should not be set
    clf_no_cov = LinearDiscriminantAnalysis(solver="svd", store_covariance=False)
    clf_no_cov.partial_fit(X_full, y_full, classes=classes)
    assert not hasattr(clf_no_cov, "covariance_")


@pytest.mark.parametrize("solver", ["eigen", "lsqr", "svd"])
def test_lda_partial_fit_after_fit_one_class_chunk(solver):
    """fit() then partial_fit() with one-class chunk must not crash."""
    from sklearn.exceptions import NotFittedError

    X_full, y_full = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    classes = np.unique(y_full)

    clf = LinearDiscriminantAnalysis(solver=solver)
    clf.fit(X_full, y_full)

    # partial_fit with only one class => reinit + early return => unfitted
    X_one = X_full[:5]
    y_one = np.full(5, classes[0])
    clf.partial_fit(X_one, y_one)

    with pytest.raises(NotFittedError):
        clf.predict(X_full)


@pytest.mark.parametrize("solver", ["eigen", "lsqr", "svd"])
def test_lda_partial_fit_multiclass_not_fitted_until_all_seen(solver):
    """Multiclass partial_fit: unfitted until all declared classes observed."""
    from sklearn.exceptions import NotFittedError

    rng = np.random.RandomState(42)
    n_features = 4
    classes = np.array([0, 1, 2])

    clf = LinearDiscriminantAnalysis(solver=solver)

    # Feed only classes 0 and 1
    X_01 = rng.randn(40, n_features)
    y_01 = np.array([0, 1] * 20)
    clf.partial_fit(X_01, y_01, classes=classes)

    with pytest.raises(NotFittedError):
        clf.predict(X_01)

    # Now feed class 2 => should become fitted
    X_2 = rng.randn(20, n_features) + 3
    y_2 = np.full(20, 2)
    clf.partial_fit(X_2, y_2)

    predictions = clf.predict(X_01)
    assert predictions.shape == (40,)


def test_lda_svd_partial_fit_scale_invariance():
    """SVD partial_fit handles features with very different scales."""
    rng = np.random.RandomState(42)
    n_samples = 300
    # Features at vastly different scales
    X = np.column_stack(
        [
            rng.randn(n_samples) * 1.0,
            rng.randn(n_samples) * 1e-4,
            rng.randn(n_samples) * 1e-6,
        ]
    )
    y = (X[:, 0] > 0).astype(int)

    clf_batch = LinearDiscriminantAnalysis(solver="svd")
    clf_batch.fit(X, y)

    clf_online = LinearDiscriminantAnalysis(solver="svd")
    classes = np.unique(y)
    chunk_size = 50
    for i in range(0, n_samples, chunk_size):
        clf_online.partial_fit(
            X[i : i + chunk_size],
            y[i : i + chunk_size],
            classes=classes if i == 0 else None,
        )

    # Predictions must match exactly
    assert_array_equal(clf_online.predict(X), clf_batch.predict(X))

    # Probabilities must be near-identical
    proba_batch = clf_batch.predict_proba(X)
    proba_online = clf_online.predict_proba(X)
    assert_allclose(proba_online, proba_batch, atol=1e-10)


def test_lda_svd_partial_fit_near_zero_variance():
    """SVD partial_fit handles low-variance and constant features."""
    rng = np.random.RandomState(42)
    n_samples = 200
    X = np.column_stack(
        [
            rng.randn(n_samples),  # informative
            rng.randn(n_samples) * 1e-8,  # low variance
            np.ones(n_samples) * 5.0,  # constant (exact zero variance)
        ]
    )
    y = (X[:, 0] > 0).astype(int)

    clf_batch = LinearDiscriminantAnalysis(solver="svd")
    clf_batch.fit(X, y)

    clf_online = LinearDiscriminantAnalysis(solver="svd")
    classes = np.unique(y)
    chunk_size = 50
    for i in range(0, n_samples, chunk_size):
        clf_online.partial_fit(
            X[i : i + chunk_size],
            y[i : i + chunk_size],
            classes=classes if i == 0 else None,
        )

    # Coefficients must be finite
    assert np.all(np.isfinite(clf_online.coef_))
    assert np.all(np.isfinite(clf_batch.coef_))

    # Predictions must match batch exactly
    assert_array_equal(clf_online.predict(X), clf_batch.predict(X))

    # Probabilities must be near-identical
    proba_batch = clf_batch.predict_proba(X)
    proba_online = clf_online.predict_proba(X)
    assert_allclose(proba_online, proba_batch, atol=1e-10)


def test_lda_svd_partial_fit_tiny_nonzero_noise():
    """Regression: tiny nonzero noise columns must match batch exactly."""
    rng = np.random.RandomState(42)
    n_samples = 200
    X = np.column_stack(
        [
            rng.randn(n_samples),  # informative
            rng.randn(n_samples) * 1e-8,  # tiny nonzero noise
            rng.randn(n_samples) * 1e-9,  # even tinier noise
        ]
    )
    y = (X[:, 0] > 0).astype(int)

    clf_batch = LinearDiscriminantAnalysis(solver="svd")
    clf_batch.fit(X, y)

    clf_online = LinearDiscriminantAnalysis(solver="svd")
    classes = np.unique(y)
    chunk_size = 40
    for i in range(0, n_samples, chunk_size):
        clf_online.partial_fit(
            X[i : i + chunk_size],
            y[i : i + chunk_size],
            classes=classes if i == 0 else None,
        )

    # Exact prediction parity
    assert_array_equal(clf_online.predict(X), clf_batch.predict(X))

    # Near-exact probability parity
    proba_batch = clf_batch.predict_proba(X)
    proba_online = clf_online.predict_proba(X)
    assert_allclose(proba_online, proba_batch, atol=1e-10)


def test_lda_svd_partial_fit_multiclass_constant_columns():
    """Regression: 3-class data with exactly-constant columns must match batch."""
    rng = np.random.RandomState(42)
    n_per_class = 50
    n_samples = 3 * n_per_class

    # 3 informative features + 2 exactly constant columns
    X_informative = rng.randn(n_samples, 3)
    X_constant = np.full((n_samples, 2), [3.14, -2.71])
    X = np.column_stack([X_informative, X_constant])
    y = np.repeat([0, 1, 2], n_per_class)

    # Shuffle with fixed seed for adversarial chunk ordering
    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]

    clf_batch = LinearDiscriminantAnalysis(solver="svd")
    clf_batch.fit(X, y)

    clf_online = LinearDiscriminantAnalysis(solver="svd")
    classes = np.unique(y)
    # chunk_size=1: worst case for accumulation drift
    for i in range(n_samples):
        clf_online.partial_fit(
            X[i : i + 1],
            y[i : i + 1],
            classes=classes if i == 0 else None,
        )

    # Exact prediction parity
    assert_array_equal(clf_online.predict(X), clf_batch.predict(X))

    # Near-exact probability parity
    proba_batch = clf_batch.predict_proba(X)
    proba_online = clf_online.predict_proba(X)
    assert_allclose(proba_online, proba_batch, atol=1e-10)


def test_lda_svd_partial_fit_multiclass_constant_columns_make_classification():
    """Regression: make_classification data with appended constant columns.

    Streaming SVD accumulation can produce near-zero (but non-zero) std for
    truly constant columns due to floating-point noise. This must be clamped
    identically to the batch path to avoid coefficient blow-ups.
    """
    X, y = make_classification(
        n_samples=700,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    # Append 2 exactly constant columns
    X = np.column_stack([X, np.full((700, 2), [3.14, -2.71])])

    clf_batch = LinearDiscriminantAnalysis(solver="svd")
    clf_batch.fit(X, y)

    classes = np.unique(y)
    for chunk_size in [1, 2, 7, 64, 700]:
        clf_online = LinearDiscriminantAnalysis(solver="svd")
        for start in range(0, len(X), chunk_size):
            end = start + chunk_size
            clf_online.partial_fit(
                X[start:end],
                y[start:end],
                classes=classes if start == 0 else None,
            )

        preds_batch = clf_batch.predict(X)
        preds_online = clf_online.predict(X)
        assert_array_equal(
            preds_online,
            preds_batch,
            err_msg=f"chunk_size={chunk_size}: prediction mismatch",
        )

        proba_batch = clf_batch.predict_proba(X)
        proba_online = clf_online.predict_proba(X)
        assert_allclose(
            proba_online,
            proba_batch,
            atol=1e-10,
            err_msg=f"chunk_size={chunk_size}: probability mismatch",
        )


@pytest.mark.parametrize("random_state", [0, 7, 11, 16])
def test_lda_svd_partial_fit_constant_columns_seed_sweep_stability(
    random_state,
):
    """Chunking should be stable on adversarial constant-column datasets."""
    X, y = make_classification(
        n_samples=700,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    X = np.column_stack([X, np.full((700, 2), [3.14, -2.71])])

    # Fixed shuffle makes this adversarial and reproducible.
    rng = np.random.RandomState(0)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]

    clf_batch = LinearDiscriminantAnalysis(solver="svd")
    clf_batch.fit(X, y)
    preds_batch = clf_batch.predict(X)
    proba_batch = clf_batch.predict_proba(X)
    acc_batch = clf_batch.score(X, y)

    classes = np.unique(y)
    chunk_sizes = [1, 2, 7, 64, 700]
    preds_by_chunk = {}
    proba_by_chunk = {}
    acc_by_chunk = {}
    for chunk_size in chunk_sizes:
        clf_online = LinearDiscriminantAnalysis(solver="svd")
        for start in range(0, len(X), chunk_size):
            end = start + chunk_size
            clf_online.partial_fit(
                X[start:end],
                y[start:end],
                classes=classes if start == 0 else None,
            )

        preds_by_chunk[chunk_size] = clf_online.predict(X)
        proba_by_chunk[chunk_size] = clf_online.predict_proba(X)
        acc_by_chunk[chunk_size] = clf_online.score(X, y)

    baseline_chunk = chunk_sizes[0]
    for chunk_size in chunk_sizes[1:]:
        assert_array_equal(
            preds_by_chunk[chunk_size],
            preds_by_chunk[baseline_chunk],
            err_msg=f"chunk_size={chunk_size}: streaming/chunking instability",
        )

    if random_state == 0:
        # Keep one strict parity guard with known-stable seed.
        assert_array_equal(preds_by_chunk[1], preds_batch)
        assert_allclose(proba_by_chunk[1], proba_batch, atol=1e-10)
    else:
        agreement = np.mean(preds_by_chunk[1] == preds_batch)
        assert agreement >= 0.87
        assert acc_by_chunk[1] >= acc_batch - 0.01
        assert acc_by_chunk[1] > 0.75


@pytest.mark.parametrize("random_state", [0, 7, 11, 16])
def test_lda_svd_partial_fit_within_std_matches_batch_to_roundoff(
    random_state,
):
    """Streaming within-class std should match batch up to roundoff."""
    X, y = make_classification(
        n_samples=700,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    X = np.column_stack([X, np.full((700, 2), [3.14, -2.71])])

    rng = np.random.RandomState(0)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]

    clf_batch = LinearDiscriminantAnalysis(solver="svd")
    clf_batch.fit(X, y)

    classes = np.unique(y)
    clf_online = LinearDiscriminantAnalysis(solver="svd")
    for i in range(len(X)):
        clf_online.partial_fit(
            X[i : i + 1],
            y[i : i + 1],
            classes=classes if i == 0 else None,
        )

    Xc = []
    for idx, group in enumerate(clf_batch.classes_):
        Xg = X[y == group]
        Xc.append(Xg - clf_batch.means_[idx, :])
    Xc = np.concatenate(Xc, axis=0)
    std_batch = np.std(Xc, axis=0)

    N_total = clf_online._class_counts.sum()
    std_online = clf_online._svd_std_from_within_sum_sq(
        clf_online._within_sum_sq, N_total
    )
    assert_allclose(std_online, std_batch, rtol=1e-12, atol=1e-12)


def test_lda_svd_partial_fit_mnist_accuracy_parity():
    """Regression: streaming SVD must match batch accuracy on MNIST_784.

    MNIST has 784 features with many near-zero-variance columns (corner pixels).
    Before the _clamp_svd_std fix, streaming produced ~30% accuracy vs ~86% batch.
    This test ensures the fix is never regressed.

    Requires SKLEARN_SKIP_NETWORK_TESTS=0 to run (auto-skipped otherwise).
    """
    import os

    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import StratifiedShuffleSplit

    if os.environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") != "0":
        pytest.skip("Set SKLEARN_SKIP_NETWORK_TESTS=0 to run this test")

    try:
        mnist = fetch_openml(
            name="mnist_784",
            version=1,
            as_frame=False,
            parser="auto",
        )
    except Exception as exc:
        pytest.skip(f"Could not fetch MNIST: {exc}")

    X_all = np.asarray(mnist.data, dtype=np.float64)
    y_all = np.asarray(mnist.target, dtype=np.int64)

    # Subsample for speed while preserving all 10 classes
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_all), size=5000, replace=False)
    X_all, y_all = X_all[idx], y_all[idx]

    # Train/test split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X_all, y_all))
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    # Batch baseline
    clf_batch = LinearDiscriminantAnalysis(solver="svd")
    clf_batch.fit(X_train, y_train)
    acc_batch = clf_batch.score(X_test, y_test)

    classes = np.unique(y_train)
    y_pred_batch = clf_batch.predict(X_test)
    for chunk_size in [256, 512, 1024, 2048]:
        clf_stream = LinearDiscriminantAnalysis(solver="svd")
        for start in range(0, len(X_train), chunk_size):
            end = min(start + chunk_size, len(X_train))
            clf_stream.partial_fit(
                X_train[start:end],
                y_train[start:end],
                classes=classes if start == 0 else None,
            )

        y_pred_stream = clf_stream.predict(X_test)
        acc_stream = np.mean(y_pred_stream == y_test)
        pred_agreement = np.mean(y_pred_stream == y_pred_batch)

        # Streaming must stay within 1% of batch.
        assert acc_stream >= acc_batch - 0.01, (
            f"chunk_size={chunk_size}: streaming accuracy {acc_stream:.4f} "
            f"is too far below batch {acc_batch:.4f}"
        )
        # Streaming should be almost prediction-identical to batch on MNIST.
        assert pred_agreement >= 0.999, (
            f"chunk_size={chunk_size}: prediction agreement {pred_agreement:.4f} "
            "is below 0.999"
        )
        # Absolute floor — catches catastrophic failure
        assert acc_stream > 0.75, (
            f"chunk_size={chunk_size}: streaming accuracy {acc_stream:.4f} "
            f"is catastrophically low"
        )
