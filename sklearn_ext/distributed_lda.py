"""Distributed merging of independently partial_fit'd LDA models (SVD path).

This module provides :func:`merge_lda_models`, which combines N
``LinearDiscriminantAnalysis`` estimators — each fitted via ``partial_fit``
with ``solver='svd'`` — into a single model whose sufficient statistics
equal the result of sequential ``partial_fit`` on all data combined.

The merge exploits the associativity and commutativity of Chan's parallel
variance algorithm, performing a single SVD on the concatenation of all
compact factors and mean-shift corrections.
"""

import math
import warnings

import numpy as np
import scipy.linalg

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils._array_api import _is_numpy_namespace, get_namespace, device


def merge_lda_models(estimators):
    """Merge N independently partial_fit'd LDA models into one.

    Each estimator must have been fitted via ``partial_fit`` with
    ``solver='svd'``. Workers may have seen different subsets of
    classes — the merge uses the union of all observed classes.

    Parameters
    ----------
    estimators : list of LinearDiscriminantAnalysis
        Two or more models fitted via partial_fit (solver='svd').

    Returns
    -------
    merged : LinearDiscriminantAnalysis
        New model whose sufficient statistics equal the merge of all
        inputs. Prediction attributes are lazily reconstructed on
        first access.

    Raises
    ------
    ValueError
        If fewer than 2 estimators, incompatible solvers, mismatched
        features, or missing streaming state (e.g. models fitted via
        ``fit()`` instead of ``partial_fit()``).
    """
    # ------------------------------------------------------------------
    # Step 1: Validation
    # ------------------------------------------------------------------
    if len(estimators) < 2:
        raise ValueError(
            "merge_lda_models requires at least 2 estimators, "
            f"got {len(estimators)}."
        )

    for i, est in enumerate(estimators):
        if not isinstance(est, LinearDiscriminantAnalysis):
            raise TypeError(
                f"estimators[{i}] is not a LinearDiscriminantAnalysis instance."
            )
        if est.solver != "svd":
            raise ValueError(
                f"estimators[{i}] has solver='{est.solver}', expected 'svd'."
            )

    _REQUIRED_ATTRS = (
        "_unscaled_S",
        "_unscaled_Vt",
        "_class_counts",
        "means_",
        "_within_sum_sq",
    )
    for i, est in enumerate(estimators):
        for attr in _REQUIRED_ATTRS:
            if not hasattr(est, attr):
                raise ValueError(
                    f"estimators[{i}] is missing attribute '{attr}'. "
                    "Ensure the model was fitted via partial_fit(), not fit()."
                )

    ref = estimators[0]
    n_features = ref.n_features_in_
    for i, est in enumerate(estimators[1:], 1):
        if est.n_features_in_ != n_features:
            raise ValueError(
                f"estimators[{i}] has n_features_in_={est.n_features_in_}, "
                f"expected {n_features}."
            )

    # Validate matching hyperparameters
    _HYPERPARAMS = ("tol", "n_components", "store_covariance", "shrinkage")
    for hp in _HYPERPARAMS:
        ref_val = getattr(ref, hp)
        for i, est in enumerate(estimators[1:], 1):
            est_val = getattr(est, hp)
            if ref_val != est_val:
                raise ValueError(
                    f"estimators[{i}] has {hp}={est_val!r}, "
                    f"expected {ref_val!r} (from estimators[0])."
                )

    # Validate matching priors (array comparison)
    ref_priors = ref.priors
    for i, est in enumerate(estimators[1:], 1):
        est_priors = est.priors
        if ref_priors is None and est_priors is None:
            continue
        if ref_priors is None or est_priors is None:
            raise ValueError(
                f"estimators[{i}] has priors={est_priors!r}, "
                f"expected {ref_priors!r} (from estimators[0])."
            )
        if not np.array_equal(np.asarray(ref_priors), np.asarray(est_priors)):
            raise ValueError(
                f"estimators[{i}] has different priors than estimators[0]."
            )

    # Array namespace / device
    xp = getattr(ref, "_array_ns", np)
    dev = getattr(ref, "_array_device", None)

    # ------------------------------------------------------------------
    # Step 2: Compute union classes
    # ------------------------------------------------------------------
    union_classes = np.unique(np.concatenate([
        np.asarray(est.classes_) for est in estimators
    ]))
    n_union = len(union_classes)

    # Build per-estimator class-index mappings
    # est_maps[i] maps union-class index -> estimator-local index (or -1)
    est_maps = []
    for est in estimators:
        local_classes = np.asarray(est.classes_)
        mapping = np.full(n_union, -1, dtype=np.intp)
        for u_idx, c in enumerate(union_classes):
            loc = np.where(local_classes == c)[0]
            if len(loc) > 0:
                mapping[u_idx] = loc[0]
        est_maps.append(mapping)

    # ------------------------------------------------------------------
    # Step 3: Remap and merge scalar statistics
    # ------------------------------------------------------------------
    acc_counts = xp.zeros(n_union, dtype=xp.float64, device=dev)
    acc_means = xp.zeros((n_union, n_features), dtype=xp.float64, device=dev)
    acc_within_ss = xp.zeros(n_features, dtype=xp.float64, device=dev)

    mean_shift_rows = []

    for est_i, est in enumerate(estimators):
        mapping = est_maps[est_i]
        est_counts = est._class_counts
        est_means = est.means_
        est_wss = est._within_sum_sq

        # Add within-class sum-of-squares from this estimator
        acc_within_ss = acc_within_ss + est_wss

        for u_idx in range(n_union):
            loc_idx = int(mapping[u_idx])
            if loc_idx < 0:
                continue

            N_B = float(est_counts[loc_idx])
            if N_B == 0:
                continue

            N_A = float(acc_counts[u_idx])
            N_new = N_A + N_B

            delta = est_means[loc_idx] - acc_means[u_idx]

            if N_A > 0:
                # Chan's parallel merge: cross-term correction
                cross_term = (N_A * N_B / N_new) * delta ** 2
                acc_within_ss = acc_within_ss + cross_term

                # Mean-shift correction row for SVD merge
                weight = math.sqrt(N_A * N_B / N_new)
                mean_shift_rows.append(weight * delta)

            # Update running mean
            acc_means[u_idx] = acc_means[u_idx] + (N_B / N_new) * delta
            acc_counts[u_idx] = xp.asarray(N_new, dtype=xp.float64, device=dev)

    # ------------------------------------------------------------------
    # Step 4: Single-pass SVD merge
    # ------------------------------------------------------------------
    Z_parts = []
    for est in estimators:
        if est._unscaled_S.shape[0] > 0:
            Z_parts.append(
                xp.expand_dims(est._unscaled_S, axis=1) * est._unscaled_Vt
            )
    Z_parts.extend(
        xp.expand_dims(xp.astype(row, xp.float64), axis=0)
        for row in mean_shift_rows
    )

    if Z_parts:
        Z = xp.concat(Z_parts, axis=0)
    else:
        Z = xp.zeros((0, n_features), dtype=xp.float64, device=dev)

    # SVD of block matrix
    if Z.shape[0] > 0:
        if _is_numpy_namespace(xp):
            _, S_new, Vt_new = scipy.linalg.svd(
                Z, full_matrices=False, check_finite=False
            )
        else:
            _, S_new, Vt_new = xp.linalg.svd(Z, full_matrices=False)

        # Truncate negligible components (same logic as _partial_fit_svd)
        eps = float(xp.finfo(Z.dtype).eps)
        threshold = eps * max(Z.shape) * float(S_new[0])
        rank = 0
        for i in range(S_new.shape[0]):
            if float(S_new[i]) > threshold:
                rank = i + 1
            else:
                break
        S_merged = S_new[:rank]
        Vt_merged = Vt_new[:rank]
    else:
        S_merged = xp.zeros(0, dtype=xp.float64, device=dev)
        Vt_merged = xp.zeros((0, n_features), dtype=xp.float64, device=dev)

    # ------------------------------------------------------------------
    # Step 5: Construct merged model
    # ------------------------------------------------------------------
    merged = LinearDiscriminantAnalysis(
        solver="svd",
        shrinkage=ref.shrinkage,
        priors=ref.priors,
        n_components=ref.n_components,
        store_covariance=ref.store_covariance,
        tol=ref.tol,
    )

    # Transfer the union classes (as xp array on device)
    merged.classes_ = xp.asarray(union_classes, device=dev)
    merged.n_features_in_ = n_features
    merged._class_counts = acc_counts
    merged.means_ = acc_means
    merged._within_sum_sq = acc_within_ss
    merged._unscaled_S = S_merged
    merged._unscaled_Vt = Vt_merged
    merged._array_ns = xp
    merged._array_device = dev

    # Copy feature_names_in_ if present on all estimators
    if hasattr(ref, "feature_names_in_"):
        merged.feature_names_in_ = ref.feature_names_in_

    # --- Early return if not all classes seen or insufficient df ---
    n_classes_seen = int(
        xp.sum(xp.astype(acc_counts > 0, xp.int32))
    )
    N_total = float(xp.sum(acc_counts))
    if n_classes_seen < n_union or N_total <= n_classes_seen:
        # Model stays in partial/unfitted state
        merged._svd_attrs_stale = False
        return merged

    # --- Derive priors ---
    if ref.priors is not None:
        merged.priors_ = xp.asarray(
            ref.priors, dtype=xp.float64, device=dev
        )
        if xp.any(merged.priors_ < 0):
            raise ValueError("priors must be non-negative")
        if abs(float(xp.sum(merged.priors_)) - 1.0) > 1e-5:
            warnings.warn(
                "The priors do not sum to 1. Renormalizing",
                UserWarning,
            )
            merged.priors_ = merged.priors_ / xp.sum(merged.priors_)
    else:
        merged.priors_ = acc_counts / N_total

    # --- Maximum number of components ---
    max_components = min(n_union - 1, n_features)
    if ref.n_components is None:
        merged._max_components = max_components
    else:
        if ref.n_components > max_components:
            raise ValueError(
                "n_components cannot be larger than "
                "min(n_features, n_classes - 1)."
            )
        merged._max_components = ref.n_components
    merged._n_features_out = merged._max_components

    # Mark prediction attributes as stale → lazy reconstruction
    merged._invalidate_svd_attrs()

    return merged
