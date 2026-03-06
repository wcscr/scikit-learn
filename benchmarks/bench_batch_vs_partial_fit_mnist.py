#!/usr/bin/env python
"""Focused benchmark: batch fit() vs single-call partial_fit() on MNIST.

Uses 10-fold stratified CV so each method is timed 10 times on the same
training splits. chunk_size = len(X_train) so partial_fit sees ALL data
in one call — no incremental merging. This isolates the code-path
overhead difference between fit() and partial_fit().
"""

import numpy as np
from time import perf_counter
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold


def main():
    print("Fetching MNIST from OpenML...")
    data = fetch_openml(name="mnist_784", version=1, as_frame=False, parser="auto")
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.int64)
    print(f"Loaded: {X.shape}, {len(np.unique(y))} classes\n")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    classes = np.unique(y)

    batch_times = []
    stream_times = []
    batch_accs = []
    stream_accs = []
    agreements = []

    print(f"{'Fold':>4}  {'t_batch':>8}  {'t_stream':>9}  {'acc_bat':>8}  "
          f"{'acc_str':>8}  {'agree':>8}  {'ratio':>7}")
    print("-" * 65)

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # --- batch fit ---
        clf_batch = LinearDiscriminantAnalysis(solver="svd")
        t0 = perf_counter()
        clf_batch.fit(X_train, y_train)
        t_batch = perf_counter() - t0

        pred_batch = clf_batch.predict(X_test)
        acc_batch = float(np.mean(pred_batch == y_test))

        # --- single-call partial_fit (full training set in one chunk) ---
        clf_stream = LinearDiscriminantAnalysis(solver="svd")
        t0 = perf_counter()
        clf_stream.partial_fit(X_train, y_train, classes=classes)
        t_stream = perf_counter() - t0

        pred_stream = clf_stream.predict(X_test)
        acc_stream = float(np.mean(pred_stream == y_test))

        agree = float(np.mean(pred_batch == pred_stream))

        batch_times.append(t_batch)
        stream_times.append(t_stream)
        batch_accs.append(acc_batch)
        stream_accs.append(acc_stream)
        agreements.append(agree)

        ratio = t_stream / t_batch
        print(f"{fold_i:>4}  {t_batch:>8.3f}  {t_stream:>9.3f}  {acc_batch:>8.4f}  "
              f"{acc_stream:>8.4f}  {agree:>8.4f}  {ratio:>7.3f}x")

    print("-" * 65)

    mean_batch = np.mean(batch_times)
    mean_stream = np.mean(stream_times)
    std_batch = np.std(batch_times)
    std_stream = np.std(stream_times)
    mean_ratio = mean_stream / mean_batch

    print(f"\n{'SUMMARY (10-fold)':>20}")
    print(f"  batch  fit():       {mean_batch:.3f}s +/- {std_batch:.3f}s")
    print(f"  partial_fit():      {mean_stream:.3f}s +/- {std_stream:.3f}s")
    print(f"  mean ratio:         {mean_ratio:.3f}x")
    print(f"  mean acc batch:     {np.mean(batch_accs):.4f}")
    print(f"  mean acc stream:    {np.mean(stream_accs):.4f}")
    print(f"  mean agreement:     {np.mean(agreements):.4f}")

    # Paired t-test
    diffs = np.array(stream_times) - np.array(batch_times)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    t_stat = mean_diff / (std_diff / np.sqrt(len(diffs)))
    print(f"\n  paired diff:        {mean_diff:+.3f}s +/- {std_diff:.3f}s")
    print(f"  t-statistic:        {t_stat:.2f}  (df=9)")
    print(f"  (positive t => partial_fit is slower)")


if __name__ == "__main__":
    main()
