"""Linear and quadratic discriminant analysis."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from numbers import Integral, Real

import numpy as np
import scipy.linalg
from scipy import linalg

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.covariance import empirical_covariance, ledoit_wolf, shrunk_covariance
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils._array_api import _expit, device, get_namespace, size
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import (
    _check_partial_fit_first_call,
    check_classification_targets,
    unique_labels,
)
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis"]


def _cov(X, shrinkage=None, covariance_estimator=None):
    """Estimate covariance matrix (using optional covariance_estimator).
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    shrinkage : {'empirical', 'auto'} or float, default=None
        Shrinkage parameter, possible values:
          - None or 'empirical': no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        Shrinkage parameter is ignored if  `covariance_estimator`
        is not None.

    covariance_estimator : estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying on the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in :mod:`sklearn.covariance``.
        If None the shrinkage parameter drives the estimate.

        .. versionadded:: 0.24

    Returns
    -------
    s : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.
    """
    if covariance_estimator is None:
        shrinkage = "empirical" if shrinkage is None else shrinkage
        if isinstance(shrinkage, str):
            if shrinkage == "auto":
                sc = StandardScaler()  # standardize features
                X = sc.fit_transform(X)
                s = ledoit_wolf(X)[0]
                # rescale
                s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            elif shrinkage == "empirical":
                s = empirical_covariance(X)
        elif isinstance(shrinkage, Real):
            s = shrunk_covariance(empirical_covariance(X), shrinkage)
    else:
        if shrinkage is not None and shrinkage != 0:
            raise ValueError(
                "covariance_estimator and shrinkage parameters "
                "are not None. Only one of the two can be set."
            )
        covariance_estimator.fit(X)
        if not hasattr(covariance_estimator, "covariance_"):
            raise ValueError(
                "%s does not have a covariance_ attribute"
                % covariance_estimator.__class__.__name__
            )
        s = covariance_estimator.covariance_
    return s


def _class_means(X, y):
    """Compute class means.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    """
    xp, is_array_api_compliant = get_namespace(X)
    classes, y = xp.unique_inverse(y)
    means = xp.zeros((classes.shape[0], X.shape[1]), device=device(X), dtype=X.dtype)

    if is_array_api_compliant:
        for i in range(classes.shape[0]):
            means[i, :] = xp.mean(X[y == i], axis=0)
    else:
        # TODO: Explore the choice of using bincount + add.at as it seems sub optimal
        # from a performance-wise
        cnt = np.bincount(y)
        np.add.at(means, y, X)
        means /= cnt[:, None]
    return means


def _class_cov(X, y, priors, shrinkage=None, covariance_estimator=None):
    """Compute weighted within-class covariance matrix.

    The per-class covariance are weighted by the class priors.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    priors : array-like of shape (n_classes,)
        Class priors.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        Shrinkage parameter is ignored if `covariance_estimator` is not None.

    covariance_estimator : estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in sklearn.covariance.
        If None, the shrinkage parameter drives the estimate.

        .. versionadded:: 0.24

    Returns
    -------
    cov : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix
    """
    classes = np.unique(y)
    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        cov += priors[idx] * np.atleast_2d(_cov(Xg, shrinkage, covariance_estimator))
    return cov


class DiscriminantAnalysisPredictionMixin:
    """Mixin class for QuadraticDiscriminantAnalysis and NearestCentroid."""

    def decision_function(self, X):
        """Apply decision function to an array of samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array of samples (test vectors).

        Returns
        -------
        y_scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values related to each class, per sample.
            In the two-class case, the shape is `(n_samples,)`, giving the
            log likelihood ratio of the positive class.
        """
        y_scores = self._decision_function(X)
        if len(self.classes_) == 2:
            return y_scores[:, 1] - y_scores[:, 0]
        return y_scores

    def predict(self, X):
        """Perform classification on an array of vectors `X`.

        Returns the class label for each sample.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class label for each sample.
        """
        scores = self._decision_function(X)
        return self.classes_.take(scores.argmax(axis=1))

    def predict_proba(self, X):
        """Estimate class probabilities.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Probability estimate of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X):
        """Estimate log class probabilities.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_log_proba : ndarray of shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        scores = self._decision_function(X)
        log_likelihood = scores - scores.max(axis=1)[:, np.newaxis]
        return log_likelihood - np.log(
            np.exp(log_likelihood).sum(axis=1)[:, np.newaxis]
        )


class LinearDiscriminantAnalysis(
    ClassNamePrefixFeaturesOutMixin,
    LinearClassifierMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Linear Discriminant Analysis.

    A classifier with a linear decision boundary, generated by fitting class
    conditional densities to the data and using Bayes' rule.

    The model fits a Gaussian density to each class, assuming that all classes
    share the same covariance matrix.

    The fitted model can also be used to reduce the dimensionality of the input
    by projecting it to the most discriminative directions, using the
    `transform` method.

    .. versionadded:: 0.17

    For a comparison between
    :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
    and :class:`~sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`, see
    :ref:`sphx_glr_auto_examples_classification_plot_lda_qda.py`.

    Read more in the :ref:`User Guide <lda_qda>`.

    Parameters
    ----------
    solver : {'svd', 'lsqr', 'eigen'}, default='svd'
        Solver to use, possible values:
          - 'svd': Singular value decomposition (default).
            Does not compute the covariance matrix, therefore this solver is
            recommended for data with a large number of features.
          - 'lsqr': Least squares solution.
            Can be combined with shrinkage or custom covariance estimator.
          - 'eigen': Eigenvalue decomposition.
            Can be combined with shrinkage or custom covariance estimator.

        .. versionchanged:: 1.2
            `solver="svd"` now has experimental Array API support. See the
            :ref:`Array API User Guide <array_api>` for more details.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        This should be left to None if `covariance_estimator` is used.
        Note that shrinkage works only with 'lsqr' and 'eigen' solvers.

        For a usage example, see
        :ref:`sphx_glr_auto_examples_classification_plot_lda.py`.

    priors : array-like of shape (n_classes,), default=None
        The class prior probabilities. By default, the class proportions are
        inferred from the training data.

    n_components : int, default=None
        Number of components (<= min(n_classes - 1, n_features)) for
        dimensionality reduction. If None, will be set to
        min(n_classes - 1, n_features). This parameter only affects the
        `transform` method.

        For a usage example, see
        :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_lda.py`.

    store_covariance : bool, default=False
        If True, explicitly compute the weighted within-class covariance
        matrix when solver is 'svd'. The matrix is always computed
        and stored for the other solvers.

        .. versionadded:: 0.17

    tol : float, default=1.0e-4
        Absolute threshold for a singular value of X to be considered
        significant, used to estimate the rank of X. Dimensions whose
        singular values are non-significant are discarded. Only used if
        solver is 'svd'.

        .. versionadded:: 0.17

    covariance_estimator : covariance estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying on the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in :mod:`sklearn.covariance`.
        if None the shrinkage parameter drives the estimate.

        This should be left to None if `shrinkage` is used.
        Note that `covariance_estimator` works only with 'lsqr' and 'eigen'
        solvers.

        .. versionadded:: 0.24

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_classes, n_features)
        Weight vector(s).

    intercept_ : ndarray of shape (n_classes,)
        Intercept term.

    covariance_ : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix. It corresponds to
        `sum_k prior_k * C_k` where `C_k` is the covariance matrix of the
        samples in class `k`. The `C_k` are estimated using the (potentially
        shrunk) biased estimator of covariance. If solver is 'svd', only
        exists when `store_covariance` is True.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0. Only available when eigen
        or svd solver is used.

    means_ : array-like of shape (n_classes, n_features)
        Class-wise means.

    priors_ : array-like of shape (n_classes,)
        Class priors (sum to 1).

    scalings_ : array-like of shape (rank, n_classes - 1)
        Scaling of the features in the space spanned by the class centroids.
        Only available for 'svd' and 'eigen' solvers.

    xbar_ : array-like of shape (n_features,)
        Overall mean. Only present if solver is 'svd'.

    classes_ : array-like of shape (n_classes,)
        Unique class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    QuadraticDiscriminantAnalysis : Quadratic Discriminant Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = LinearDiscriminantAnalysis()
    >>> clf.fit(X, y)
    LinearDiscriminantAnalysis()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    _parameter_constraints: dict = {
        "solver": [StrOptions({"svd", "lsqr", "eigen"})],
        "shrinkage": [StrOptions({"auto"}), Interval(Real, 0, 1, closed="both"), None],
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "priors": ["array-like", None],
        "store_covariance": ["boolean"],
        "tol": [Interval(Real, 0, None, closed="left")],
        "covariance_estimator": [HasMethods("fit"), None],
    }

    def __init__(
        self,
        solver="svd",
        shrinkage=None,
        priors=None,
        n_components=None,
        store_covariance=False,
        tol=1e-4,
        covariance_estimator=None,
    ):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance  # used only in svd solver
        self.tol = tol  # used only in svd solver
        self.covariance_estimator = covariance_estimator

    def _solve_lstsq(self, X, y, shrinkage, covariance_estimator):
        """Least squares solver.

        The least squares solver computes a straightforward solution of the
        optimal decision rule based directly on the discriminant functions. It
        can only be used for classification (with any covariance estimator),
        because
        estimation of eigenvectors is not performed. Therefore, dimensionality
        reduction with the transform is not supported.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target values.

        shrinkage : 'auto', float or None
            Shrinkage parameter, possible values:
              - None: no shrinkage.
              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
              - float between 0 and 1: fixed shrinkage parameter.

            Shrinkage parameter is ignored if  `covariance_estimator` is
            not None

        covariance_estimator : estimator, default=None
            If not None, `covariance_estimator` is used to estimate
            the covariance matrices instead of relying the empirical
            covariance estimator (with potential shrinkage).
            The object should have a fit method and a ``covariance_`` attribute
            like the estimators in sklearn.covariance.
            if None the shrinkage parameter drives the estimate.

            .. versionadded:: 0.24

        Notes
        -----
        This solver is based on [1]_, section 2.6.2, pp. 39-41.

        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(
            X, y, self.priors_, shrinkage, covariance_estimator
        )
        self._solve_lsqr_from_covariance(self.covariance_)

    def _solve_lsqr_from_covariance(self, covariance):
        """Compute coef_ and intercept_ from a precomputed covariance matrix.

        Parameters
        ----------
        covariance : ndarray of shape (n_features, n_features)
            The within-class covariance matrix.
        """
        self.coef_ = linalg.lstsq(covariance, self.means_.T)[0].T
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def _solve_eigen(self, X, y, shrinkage, covariance_estimator):
        """Eigenvalue solver.

        The eigenvalue solver computes the optimal solution of the Rayleigh
        coefficient (basically the ratio of between class scatter to within
        class scatter). This solver supports both classification and
        dimensionality reduction (with any covariance estimator).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        shrinkage : 'auto', float or None
            Shrinkage parameter, possible values:
              - None: no shrinkage.
              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
              - float between 0 and 1: fixed shrinkage constant.

            Shrinkage parameter is ignored if  `covariance_estimator` is
            not None

        covariance_estimator : estimator, default=None
            If not None, `covariance_estimator` is used to estimate
            the covariance matrices instead of relying the empirical
            covariance estimator (with potential shrinkage).
            The object should have a fit method and a ``covariance_`` attribute
            like the estimators in sklearn.covariance.
            if None the shrinkage parameter drives the estimate.

            .. versionadded:: 0.24

        Notes
        -----
        This solver is based on [1]_, section 3.8.3, pp. 121-124.

        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(
            X, y, self.priors_, shrinkage, covariance_estimator
        )

        Sw = self.covariance_  # within scatter
        St = _cov(X, shrinkage, covariance_estimator)  # total scatter
        Sb = St - Sw  # between scatter

        self._solve_eigen_from_matrices(Sb, Sw)

    def _solve_eigen_from_matrices(self, Sb, Sw):
        """Solve the generalized eigenvalue problem given scatter matrices.

        Parameters
        ----------
        Sb : ndarray of shape (n_features, n_features)
            Between-class scatter matrix.

        Sw : ndarray of shape (n_features, n_features)
            Within-class scatter matrix.
        """
        evals, evecs = linalg.eigh(Sb, Sw)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][
            : self._max_components
        ]
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def _solve_svd(self, X, y):
        """SVD solver.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        """
        xp, is_array_api_compliant = get_namespace(X)

        if is_array_api_compliant:
            svd = xp.linalg.svd
        else:
            svd = scipy.linalg.svd

        n_samples, _ = X.shape
        n_classes = self.classes_.shape[0]

        self.means_ = _class_means(X, y)
        if self.store_covariance:
            self.covariance_ = _class_cov(X, y, self.priors_)

        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group]
            Xc.append(Xg - self.means_[idx, :])

        self.xbar_ = self.priors_ @ self.means_

        Xc = xp.concat(Xc, axis=0)

        # 1) within (univariate) scaling by with classes std-dev
        std = xp.std(Xc, axis=0)
        # avoid division by zero in normalization
        std[std == 0] = 1.0
        fac = xp.asarray(1.0 / (n_samples - n_classes), dtype=X.dtype, device=device(X))

        # 2) Within variance scaling
        X = xp.sqrt(fac) * (Xc / std)
        # SVD of centered (within)scaled data
        _, S, Vt = svd(X, full_matrices=False)

        rank = xp.sum(xp.astype(S > self.tol, xp.int32))
        # Scaling of within covariance is: V' 1/S
        scalings = (Vt[:rank, :] / std).T / S[:rank]
        fac = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)

        # 3) Between variance scaling
        # Scale weighted centers
        X = (
            (xp.sqrt((n_samples * self.priors_) * fac)) * (self.means_ - self.xbar_).T
        ).T @ scalings
        # Centers are living in a space with n_classes-1 dim (maximum)
        # Use SVD to find projection in the space spanned by the
        # (n_classes) centers
        _, S, Vt = svd(X, full_matrices=False)

        if self._max_components == 0:
            self.explained_variance_ratio_ = xp.empty((0,), dtype=S.dtype)
        else:
            self.explained_variance_ratio_ = (S**2 / xp.sum(S**2))[
                : self._max_components
            ]

        rank = xp.sum(xp.astype(S > self.tol * S[0], xp.int32))
        self.scalings_ = scalings @ Vt.T[:, :rank]
        coef = (self.means_ - self.xbar_) @ self.scalings_
        self.intercept_ = -0.5 * xp.sum(coef**2, axis=1) + xp.log(self.priors_)
        self.coef_ = coef @ self.scalings_.T
        self.intercept_ -= self.xbar_ @ self.coef_.T

    def _partial_fit_svd(self, X, y, first_call):
        """Incremental SVD update for partial_fit.

        Maintains a compact SVD factorization (S, Vt) of all within-class
        centered data seen so far, avoiding O(D^2) covariance storage.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (already validated).

        y : ndarray of shape (n_samples,)
            Target values (already validated).

        first_call : bool
            Whether this is the first call to partial_fit.
        """
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        # --- Initialize streaming state ---
        if first_call or not hasattr(self, "_unscaled_S"):
            self._class_counts = np.zeros(n_classes, dtype=np.float64)
            self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
            self._unscaled_S = np.empty(0, dtype=np.float64)
            self._unscaled_Vt = np.empty(
                (0, n_features), dtype=np.float64
            )

        # --- Per-chunk: compute local stats ---
        chunk_means = np.zeros((n_classes, n_features), dtype=np.float64)
        chunk_counts = np.zeros(n_classes, dtype=np.float64)
        for idx, c in enumerate(self.classes_):
            mask = y == c
            m_k = np.sum(mask)
            if m_k > 0:
                chunk_means[idx] = np.mean(X[mask], axis=0)
                chunk_counts[idx] = m_k

        # --- Compute mean-shift correction vectors (parallel axis theorem) ---
        mean_shift_rows = []
        for idx in range(n_classes):
            N_old = self._class_counts[idx]
            N_chunk = chunk_counts[idx]
            if N_old > 0 and N_chunk > 0:
                N_new = N_old + N_chunk
                weight = np.sqrt(N_old * N_chunk / N_new)
                mean_shift_rows.append(
                    weight * (self.means_[idx] - chunk_means[idx])
                )

        # --- Update global means and counts ---
        for idx in range(n_classes):
            m_k = chunk_counts[idx]
            if m_k == 0:
                continue
            N_old = self._class_counts[idx]
            N_new = N_old + m_k
            delta = chunk_means[idx] - self.means_[idx]
            self.means_[idx] += (m_k / N_new) * delta
            self._class_counts[idx] = N_new

        # --- Center chunk by local class means ---
        Xc = np.empty_like(X)
        for idx, c in enumerate(self.classes_):
            mask = y == c
            if np.any(mask):
                Xc[mask] = X[mask] - chunk_means[idx]

        # --- Build block matrix Z ---
        blocks = []
        if self._unscaled_S.size > 0:
            blocks.append(self._unscaled_S[:, None] * self._unscaled_Vt)
        blocks.append(Xc)
        if mean_shift_rows:
            blocks.append(np.array(mean_shift_rows))
        Z = np.vstack(blocks)

        # --- SVD of block matrix ---
        _, S_new, Vt_new = scipy.linalg.svd(Z, full_matrices=False)

        # Truncate negligible components
        keep = S_new > self.tol
        self._unscaled_S = S_new[keep]
        self._unscaled_Vt = Vt_new[keep]

        # --- Early return if fewer than 2 classes seen or insufficient df ---
        n_classes_seen = np.count_nonzero(self._class_counts)
        N_total = self._class_counts.sum()
        if n_classes_seen < 2 or N_total <= n_classes_seen:
            return

        # --- Derive priors ---
        N_total = self._class_counts.sum()
        if self.priors is not None:
            self.priors_ = np.asarray(self.priors, dtype=np.float64)
            if np.any(self.priors_ < 0):
                raise ValueError("priors must be non-negative")
            if np.abs(np.sum(self.priors_) - 1.0) > 1e-5:
                warnings.warn(
                    "The priors do not sum to 1. Renormalizing",
                    UserWarning,
                )
                self.priors_ = self.priors_ / self.priors_.sum()
        else:
            self.priors_ = self._class_counts / N_total

        # --- Maximum number of components ---
        max_components = min(n_classes - 1, n_features)
        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than "
                    "min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        # --- Reconstruct public attributes from compact SVD ---
        # Reconstruct within-class std from the SVD factors
        # std[j] = sqrt(sum_i (S[i] * Vt[i,j])^2 / N_total)
        SVt = self._unscaled_S[:, None] * self._unscaled_Vt
        std = np.sqrt(np.sum(SVt**2, axis=0) / N_total)
        std[std == 0] = 1.0

        # Within-class scaling (equivalent to batch _solve_svd)
        fac = 1.0 / (N_total - n_classes_seen)
        scaled = np.sqrt(fac) * (SVt / std)
        _, S_scaled, Vt_scaled = scipy.linalg.svd(
            scaled, full_matrices=False
        )

        rank = np.sum(S_scaled > self.tol)
        if rank == 0:
            rank = 1  # ensure at least one component
        scalings = (Vt_scaled[:rank, :] / std).T / S_scaled[:rank]

        # Between-class scaling
        self.xbar_ = self.priors_ @ self.means_
        fac_between = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)
        X_between = (
            (np.sqrt((N_total * self.priors_) * fac_between))
            * (self.means_ - self.xbar_).T
        ).T @ scalings

        _, S_between, Vt_between = scipy.linalg.svd(
            X_between, full_matrices=False
        )

        if self._max_components == 0:
            self.explained_variance_ratio_ = np.empty(
                (0,), dtype=S_between.dtype
            )
        else:
            s2_sum = np.sum(S_between**2)
            if s2_sum > 0:
                self.explained_variance_ratio_ = (
                    S_between**2 / s2_sum
                )[: self._max_components]
            else:
                self.explained_variance_ratio_ = np.zeros(
                    min(len(S_between), self._max_components),
                    dtype=S_between.dtype,
                )

        if len(S_between) > 0:
            rank_between = np.sum(S_between > self.tol * S_between[0])
        else:
            rank_between = 0
        if rank_between == 0:
            rank_between = 1
        self.scalings_ = scalings @ Vt_between.T[:, :rank_between]
        coef = (self.means_ - self.xbar_) @ self.scalings_
        self.intercept_ = (
            -0.5 * np.sum(coef**2, axis=1) + np.log(self.priors_)
        )
        self.coef_ = coef @ self.scalings_.T
        self.intercept_ -= self.xbar_ @ self.coef_.T

    @_fit_context(
        # LinearDiscriminantAnalysis.covariance_estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y):
        """Fit the Linear Discriminant Analysis model.

        .. versionchanged:: 0.19
            `store_covariance` and `tol` has been moved to main constructor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        xp, _ = get_namespace(X)

        X, y = validate_data(
            self, X, y, ensure_min_samples=2, dtype=[xp.float64, xp.float32]
        )
        self.classes_ = unique_labels(y)
        n_samples, n_features = X.shape
        n_classes = self.classes_.shape[0]

        if n_samples == n_classes:
            raise ValueError(
                "The number of samples must be more than the number of classes."
            )

        if self.priors is None:  # estimate priors from sample
            _, cnts = xp.unique_counts(y)  # non-negative ints
            self.priors_ = xp.astype(cnts, X.dtype) / float(n_samples)
        else:
            self.priors_ = xp.asarray(self.priors, dtype=X.dtype)

        if xp.any(self.priors_ < 0):
            raise ValueError("priors must be non-negative")

        if xp.abs(xp.sum(self.priors_) - 1.0) > 1e-5:
            warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        # Maximum number of components no matter what n_components is
        # specified:
        max_components = min(n_classes - 1, n_features)

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        if self.solver == "svd":
            if self.shrinkage is not None:
                raise NotImplementedError("shrinkage not supported with 'svd' solver.")
            if self.covariance_estimator is not None:
                raise ValueError(
                    "covariance estimator "
                    "is not supported "
                    "with svd solver. Try another solver"
                )
            self._solve_svd(X, y)
        elif self.solver == "lsqr":
            self._solve_lstsq(
                X,
                y,
                shrinkage=self.shrinkage,
                covariance_estimator=self.covariance_estimator,
            )
        elif self.solver == "eigen":
            self._solve_eigen(
                X,
                y,
                shrinkage=self.shrinkage,
                covariance_estimator=self.covariance_estimator,
            )
        if size(self.classes_) == 2:  # treat binary case as a special case
            coef_ = xp.asarray(self.coef_[1, :] - self.coef_[0, :], dtype=X.dtype)
            self.coef_ = xp.reshape(coef_, (1, -1))
            intercept_ = xp.asarray(
                self.intercept_[1] - self.intercept_[0], dtype=X.dtype
            )
            self.intercept_ = xp.reshape(intercept_, (1,))
        self._n_features_out = self._max_components
        return self

    @_fit_context(
        # LinearDiscriminantAnalysis.covariance_estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def partial_fit(self, X, y, classes=None):
        """Incrementally fit the Linear Discriminant Analysis model.

        This method allows online learning by updating sufficient statistics
        with each new batch of data using a pairwise moment-merge update [1]_:

        ``S_new = S_old + S_chunk + (n_old * n_chunk / n_new) * outer(delta, delta)``

        where ``delta = mean_chunk - mean_old``. This is applied per class to
        accumulate the pooled within-class scatter matrix. The resulting model
        is mathematically equivalent to fitting on all data at once.

        .. versionadded:: 1.9

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all classes that can possibly appear in the `y` vector.
            Must be provided at the first call to ``partial_fit``, can be
            omitted in subsequent calls.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        The estimator is not usable for prediction until all classes have
        been observed and sufficient samples have been accumulated (at
        least ``n_classes`` total samples for the ``eigen`` solver).
        Until then, calling ``predict`` will raise ``NotFittedError``.

        References
        ----------
        .. [1] T. F. Chan, G. H. Golub, and R. J. LeVeque, "Updating formulae
               and a pairwise algorithm for computing sample variances,"
               Technical Report STAN-CS-79-773, Stanford University, 1979.

        .. [2] T. F. Chan, G. H. Golub, and R. J. LeVeque, "Algorithms for
               computing the sample variance: Analysis and recommendations,"
               The American Statistician, vol. 37, no. 3, pp. 242-247, 1983.
        """
        if self.solver != "svd":
            if self.shrinkage == "auto" or self.covariance_estimator is not None:
                raise NotImplementedError(
                    "partial_fit does not support shrinkage='auto' or a custom "
                    "covariance_estimator. Use a float shrinkage value or None."
                )

        first_call = _check_partial_fit_first_call(self, classes)

        X, y = validate_data(
            self,
            X,
            y,
            ensure_min_samples=1,
            dtype=[np.float64, np.float32],
            reset=first_call,
        )

        # Validate that y contains only known classes
        unexpected = np.setdiff1d(y, self.classes_)
        if len(unexpected) > 0:
            raise ValueError(
                f"The target label(s) {unexpected} in y do not exist "
                f"in the initial classes {self.classes_}"
            )

        # Route to solver-specific method
        if self.solver == "svd":
            self._partial_fit_svd(X, y, first_call)
        else:
            self._partial_fit_covariance(X, y, first_call)

        # Binary case: reduce to 1D output
        if hasattr(self, "coef_") and len(self.classes_) == 2:
            self.coef_ = (self.coef_[1, :] - self.coef_[0, :])[np.newaxis, :]
            self.intercept_ = np.array(
                [self.intercept_[1] - self.intercept_[0]]
            )

        if hasattr(self, "_max_components"):
            self._n_features_out = self._max_components
        return self

    def _partial_fit_covariance(self, X, y, first_call):
        """Covariance-based incremental update for partial_fit.

        Uses Chan's parallel variance update to accumulate the pooled
        within-class scatter matrix. Used by eigen and lsqr solvers.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (already validated).

        y : ndarray of shape (n_samples,)
            Target values (already validated).

        first_call : bool
            Whether this is the first call to partial_fit.
        """
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        if first_call or not hasattr(self, "_class_counts"):
            # Reinitialize accumulators on first call or after a prior
            # fit() call which does not set these private attributes.
            self._class_counts = np.zeros(n_classes, dtype=np.float64)
            self.means_ = np.zeros(
                (n_classes, n_features), dtype=np.float64
            )
            self._unscaled_covariance = np.zeros(
                (n_features, n_features), dtype=np.float64
            )

        # --- Chan's parallel variance update per class ---
        for idx, c in enumerate(self.classes_):
            X_k = X[y == c]
            m_k = X_k.shape[0]
            if m_k == 0:
                continue

            chunk_mean = np.mean(X_k, axis=0)
            if m_k == 1:
                S_chunk = np.zeros((n_features, n_features), dtype=np.float64)
            else:
                X_k_centered = X_k - chunk_mean
                S_chunk = X_k_centered.T @ X_k_centered

            N_k_old = self._class_counts[idx]
            N_k_new = N_k_old + m_k
            delta = chunk_mean - self.means_[idx]

            self._unscaled_covariance += S_chunk + (
                N_k_old * m_k / N_k_new
            ) * np.outer(delta, delta)
            self.means_[idx] += (m_k / N_k_new) * delta
            self._class_counts[idx] = N_k_new

        # --- Derive public parameters ---
        N_total = self._class_counts.sum()

        if self.priors is not None:
            self.priors_ = np.asarray(self.priors, dtype=np.float64)
            if np.any(self.priors_ < 0):
                raise ValueError("priors must be non-negative")
            if np.abs(np.sum(self.priors_) - 1.0) > 1e-5:
                warnings.warn(
                    "The priors do not sum to 1. Renormalizing",
                    UserWarning,
                )
                self.priors_ = self.priors_ / self.priors_.sum()
        else:
            self.priors_ = self._class_counts / N_total

        # Cannot solve until at least 2 classes have been observed.
        # For the eigen solver, also require enough within-class degrees
        # of freedom for a non-singular covariance matrix.
        n_classes_seen = np.count_nonzero(self._class_counts)
        within_df = N_total - n_classes_seen
        if n_classes_seen < 2 or (
            self.solver == "eigen" and within_df < n_features
        ):
            return

        # Maximum number of components
        max_components = min(n_classes - 1, n_features)
        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than "
                    "min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        covariance = self._unscaled_covariance / N_total

        # Apply explicit float shrinkage
        if self.shrinkage is not None:
            covariance = shrunk_covariance(covariance, self.shrinkage)

        self.covariance_ = covariance

        if self.solver == "eigen":
            # Reconstruct between-class scatter
            mu_global = np.average(
                self.means_, axis=0, weights=self._class_counts
            )
            Sb = sum(
                self._class_counts[i]
                * np.outer(
                    self.means_[i] - mu_global, self.means_[i] - mu_global
                )
                for i in range(n_classes)
            ) / N_total

            if self.shrinkage is not None:
                St = shrunk_covariance(
                    (self._unscaled_covariance
                     + sum(
                         self._class_counts[i]
                         * np.outer(
                             self.means_[i] - mu_global,
                             self.means_[i] - mu_global,
                         )
                         for i in range(n_classes)
                     )) / N_total,
                    self.shrinkage,
                )
                Sb = St - covariance

            self._solve_eigen_from_matrices(Sb=Sb, Sw=covariance)
        elif self.solver == "lsqr":
            self._solve_lsqr_from_covariance(covariance)

    def __sklearn_is_fitted__(self):
        """Check fitted status by verifying solver output exists."""
        return hasattr(self, "coef_")

    def transform(self, X):
        """Project data to maximize class separation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components) or \
            (n_samples, min(rank, n_components))
            Transformed data. In the case of the 'svd' solver, the shape
            is (n_samples, min(rank, n_components)).
        """
        if self.solver == "lsqr":
            raise NotImplementedError(
                "transform not implemented for 'lsqr' solver (use 'svd' or 'eigen')."
            )
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if self.solver == "svd":
            X_new = (X - self.xbar_) @ self.scalings_
        elif self.solver == "eigen":
            X_new = X @ self.scalings_

        return X_new[:, : self._max_components]

    def predict_proba(self, X):
        """Estimate probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Estimated probabilities.
        """
        check_is_fitted(self)
        xp, _ = get_namespace(X)
        decision = self.decision_function(X)
        if size(self.classes_) == 2:
            proba = _expit(decision, xp=xp)
            return xp.stack([1 - proba, proba], axis=1)
        else:
            return softmax(decision)

    def predict_log_proba(self, X):
        """Estimate log probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        xp, _ = get_namespace(X)
        prediction = self.predict_proba(X)

        smallest_normal = xp.finfo(prediction.dtype).smallest_normal
        prediction[prediction == 0.0] += smallest_normal
        return xp.log(prediction)

    def decision_function(self, X):
        """Apply decision function to an array of samples.

        The decision function is equal (up to a constant factor) to the
        log-posterior of the model, i.e. `log p(y = k | x)`. In a binary
        classification setting this instead corresponds to the difference
        `log p(y = 1 | x) - log p(y = 0 | x)`. See :ref:`lda_qda_math`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array of samples (test vectors).

        Returns
        -------
        y_scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values related to each class, per sample.
            In the two-class case, the shape is `(n_samples,)`, giving the
            log likelihood ratio of the positive class.
        """
        # Only overrides for the docstring.
        return super().decision_function(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.array_api_support = True
        return tags


class QuadraticDiscriminantAnalysis(
    DiscriminantAnalysisPredictionMixin, ClassifierMixin, BaseEstimator
):
    """Quadratic Discriminant Analysis.

    A classifier with a quadratic decision boundary, generated
    by fitting class conditional densities to the data
    and using Bayes' rule.

    The model fits a Gaussian density to each class.

    .. versionadded:: 0.17

    For a comparison between
    :class:`~sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
    and :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`, see
    :ref:`sphx_glr_auto_examples_classification_plot_lda_qda.py`.

    Read more in the :ref:`User Guide <lda_qda>`.

    Parameters
    ----------
    solver : {'svd', 'eigen'}, default='svd'
        Solver to use, possible values:
          - 'svd': Singular value decomposition (default).
            Does not compute the covariance matrix, therefore this solver is
            recommended for data with a large number of features.
          - 'eigen': Eigenvalue decomposition.
            Can be combined with shrinkage or custom covariance estimator.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

          Enabling shrinkage is expected to improve the model when some
          classes have a relatively small number of training data points
          compared to the number of features by mitigating overfitting during
          the covariance estimation step.

        This should be left to `None` if `covariance_estimator` is used.
        Note that shrinkage works only with 'eigen' solver.

    priors : array-like of shape (n_classes,), default=None
        Class priors. By default, the class proportions are inferred from the
        training data.

    reg_param : float, default=0.0
        Regularizes the per-class covariance estimates by transforming S2 as
        ``S2 = (1 - reg_param) * S2 + reg_param * np.eye(n_features)``,
        where S2 corresponds to the `scaling_` attribute of a given class.

    store_covariance : bool, default=False
        If True, the class covariance matrices are explicitly computed and
        stored in the `self.covariance_` attribute.

        .. versionadded:: 0.17

    tol : float, default=1.0e-4
        Absolute threshold for the covariance matrix to be considered rank
        deficient after applying some regularization (see `reg_param`) to each
        `Sk` where `Sk` represents covariance matrix for k-th class. This
        parameter does not affect the predictions. It controls when a warning
        is raised if the covariance matrix is not full rank.

        .. versionadded:: 0.17

    covariance_estimator : covariance estimator, default=None
        If not None, `covariance_estimator` is used to estimate the covariance
        matrices instead of relying on the empirical covariance estimator
        (with potential shrinkage).  The object should have a fit method and
        a ``covariance_`` attribute like the estimators in
        :mod:`sklearn.covariance`. If None the shrinkage parameter drives the
        estimate.

        This should be left to `None` if `shrinkage` is used.
        Note that `covariance_estimator` works only with the 'eigen' solver.

    Attributes
    ----------
    covariance_ : list of len n_classes of ndarray \
            of shape (n_features, n_features)
        For each class, gives the covariance matrix estimated using the
        samples of that class. The estimations are unbiased. Only present if
        `store_covariance` is True.

    means_ : array-like of shape (n_classes, n_features)
        Class-wise means.

    priors_ : array-like of shape (n_classes,)
        Class priors (sum to 1).

    rotations_ : list of len n_classes of ndarray of shape (n_features, n_k)
        For each class k an array of shape (n_features, n_k), where
        ``n_k = min(n_features, number of elements in class k)``
        It is the rotation of the Gaussian distribution, i.e. its
        principal axis. It corresponds to `V`, the matrix of eigenvectors
        coming from the SVD of `Xk = U S Vt` where `Xk` is the centered
        matrix of samples from class k.

    scalings_ : list of len n_classes of ndarray of shape (n_k,)
        For each class, contains the scaling of
        the Gaussian distributions along its principal axes, i.e. the
        variance in the rotated coordinate system. It corresponds to `S^2 /
        (n_samples - 1)`, where `S` is the diagonal matrix of singular values
        from the SVD of `Xk`, where `Xk` is the centered matrix of samples
        from class k.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    LinearDiscriminantAnalysis : Linear Discriminant Analysis.

    Examples
    --------
    >>> from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = QuadraticDiscriminantAnalysis()
    >>> clf.fit(X, y)
    QuadraticDiscriminantAnalysis()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    _parameter_constraints: dict = {
        "solver": [StrOptions({"svd", "eigen"})],
        "shrinkage": [StrOptions({"auto"}), Interval(Real, 0, 1, closed="both"), None],
        "priors": ["array-like", None],
        "reg_param": [Interval(Real, 0, 1, closed="both")],
        "store_covariance": ["boolean"],
        "tol": [Interval(Real, 0, None, closed="left")],
        "covariance_estimator": [HasMethods("fit"), None],
    }

    def __init__(
        self,
        *,
        solver="svd",
        shrinkage=None,
        priors=None,
        reg_param=0.0,
        store_covariance=False,
        tol=1.0e-4,
        covariance_estimator=None,
    ):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol
        self.covariance_estimator = covariance_estimator

    def _solve_eigen(self, X):
        """Eigenvalue solver.

        The eigenvalue solver uses the eigen decomposition of the data
        to compute the rotation and scaling matrices used for scoring
        new samples. This solver supports use of any covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        n_samples, n_features = X.shape

        cov = _cov(X, self.shrinkage, self.covariance_estimator)
        scaling, rotation = linalg.eigh(cov)  # scalings are eigenvalues
        rotation = rotation[:, np.argsort(scaling)[::-1]]  # sort eigenvectors
        scaling = scaling[np.argsort(scaling)[::-1]]  # sort eigenvalues
        return scaling, rotation, cov

    def _solve_svd(self, X):
        """SVD solver for Quadratic Discriminant Analysis.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        n_samples, n_features = X.shape

        mean = X.mean(0)
        Xc = X - mean
        # Xc = U * S * V.T
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        scaling = (S**2) / (n_samples - 1)  # scalings are squared singular values
        scaling = ((1 - self.reg_param) * scaling) + self.reg_param
        rotation = Vt.T

        cov = None
        if self.store_covariance:
            # cov = V * (S^2 / (n-1)) * V.T
            cov = scaling * Vt.T @ Vt

        return scaling, rotation, cov

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the model according to the given training data and parameters.

        .. versionchanged:: 0.19
            ``store_covariances`` has been moved to main constructor as
            ``store_covariance``.

        .. versionchanged:: 0.19
            ``tol`` has been moved to main constructor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values (integers).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(
                "The number of classes has to be greater than one. Got "
                f"{n_classes} class."
            )
        if self.priors is None:
            _, cnts = np.unique(y, return_counts=True)
            self.priors_ = cnts / float(n_samples)
        else:
            self.priors_ = np.array(self.priors)

        if self.solver == "svd":
            if self.shrinkage is not None:
                # Support for `shrinkage` could be implemented as in
                # https://github.com/scikit-learn/scikit-learn/issues/32590
                raise NotImplementedError("shrinkage not supported with 'svd' solver.")
            if self.covariance_estimator is not None:
                raise ValueError(
                    "covariance_estimator is not supported with solver='svd'. "
                    "Try solver='eigen' instead."
                )
            specific_solver = self._solve_svd
        elif self.solver == "eigen":
            specific_solver = self._solve_eigen

        means = []
        cov = []
        scalings = []
        rotations = []
        for class_idx, class_label in enumerate(self.classes_):
            X_class = X[y == class_label, :]
            if len(X_class) == 1:
                raise ValueError(
                    "y has only 1 sample in class %s, covariance is ill defined."
                    % str(self.classes_[class_idx])
                )

            mean_class = X_class.mean(0)
            means.append(mean_class)

            scaling_class, rotation_class, cov_class = specific_solver(X_class)

            rank = np.sum(scaling_class > self.tol)
            if rank < n_features:
                n_samples_class = X_class.shape[0]
                if self.solver == "svd" and n_samples_class <= n_features:
                    raise linalg.LinAlgError(
                        f"The covariance matrix of class {class_label} is not full "
                        f"rank. When using `solver='svd'` the number of samples in "
                        f"each class should be more than the number of features, but "
                        f"class {class_label} has {n_samples_class} samples and "
                        f"{n_features} features. Try using `solver='eigen'` and "
                        f"setting the parameter `shrinkage` for regularization."
                    )
                else:
                    msg_param = "shrinkage" if self.solver == "eigen" else "reg_param"
                    raise linalg.LinAlgError(
                        f"The covariance matrix of class {class_label} is not full "
                        f"rank. Increase the value of `{msg_param}` to reduce the "
                        f"collinearity.",
                    )

            cov.append(cov_class)
            scalings.append(scaling_class)
            rotations.append(rotation_class)

        if self.store_covariance:
            self.covariance_ = cov
        self.means_ = np.asarray(means)
        self.scalings_ = scalings
        self.rotations_ = rotations
        return self

    def _decision_function(self, X):
        # return log posterior, see eq (4.12) p. 110 of the ESL.
        check_is_fitted(self)

        X = validate_data(self, X, reset=False)
        norm2 = []
        for i in range(len(self.classes_)):
            R = self.rotations_[i]
            S = self.scalings_[i]
            Xm = X - self.means_[i]
            X2 = np.dot(Xm, R * (S ** (-0.5)))
            norm2.append(np.sum(X2**2, axis=1))
        norm2 = np.array(norm2).T  # shape = [len(X), n_classes]
        u = np.asarray([np.sum(np.log(s)) for s in self.scalings_])
        return -0.5 * (norm2 + u) + np.log(self.priors_)

    def decision_function(self, X):
        """Apply decision function to an array of samples.

        The decision function is equal (up to a constant factor) to the
        log-posterior of the model, i.e. `log p(y = k | x)`. In a binary
        classification setting this instead corresponds to the difference
        `log p(y = 1 | x) - log p(y = 0 | x)`. See :ref:`lda_qda_math`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array of samples (test vectors).

        Returns
        -------
        C : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values related to each class, per sample.
            In the two-class case, the shape is `(n_samples,)`, giving the
            log likelihood ratio of the positive class.
        """
        # Only overrides for the docstring.
        return super().decision_function(X)
