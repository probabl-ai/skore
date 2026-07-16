"""
.. _example_skd005_underrepresented_classes:

SKD005 — Underrepresented classes
=================================

This example demonstrates mitigations when check
:ref:`SKD005 <skd005-underrepresented-classes>` fires on a multiclass task.
The check flags classes that each represent less than 10 % of rows — a pattern
where overall accuracy can mask near-zero recall on rare labels.

Mitigations from the :ref:`automated_checks` user guide:

- use per-class metrics (precision, recall, F1 per class),
- resample the dataset,
- use class weights in the estimator,
- collect more data for the underrepresented classes if possible.

We take a 10,000-row stratified subsample of the Covertype dataset so rare
forest types fall below the threshold. The goal is to monitor per-class
performance and rebalance labels when equal representation is required.
"""

# %%
# Load the Covertype dataset
# ==========================
#
# The full Covertype task has seven forest types. A small stratified subsample
# keeps frequent classes well represented while types 3–7 drop below 10 % each.

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

df = fetch_covtype(as_frame=True).frame
y_full = df["Cover_Type"].astype(str)
X_full = df.drop(columns=["Cover_Type"])

X, X_holdout, y, y_holdout = train_test_split(
    X_full,
    y_full,
    train_size=10_000,
    stratify=y_full,
    random_state=42,
)
y = pd.Series(y, name="class")
y_holdout = pd.Series(y_holdout, name="class")

# %%
# Inspect the features matrix with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
# The class histogram shows that the classes are not evenly distributed and
# types 3-7 will trigger SKD005.

TableReport(y)

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from skore import TrainTestSplit, evaluate

splitter = TrainTestSplit(
    test_size=0.2,
    random_state=42,
    stratify=y,
)

multiclass_metrics = ["accuracy", "precision", "recall"]
classifier = HistGradientBoostingClassifier(random_state=42)

# %%
# Trigger SKD005 — default classifier on imbalanced classes
# =========================================================
#
# A default gradient boosting classifier does not change label counts. Run
# :meth:`~skore.EstimatorReport.checks.summarize` with ``fast_mode=True`` to
# skip unrelated slow checks, then inspect per-class metrics below.

report = evaluate(
    classifier,
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# SKD005 should list classes 3, 4, 5, 6, and 7 as under 10 % of rows.

report.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Monitor per-class metrics
# =========================
#
# Accuracy alone can look strong when frequent classes dominate the table. Here,
# types 1 and 2 make up most rows, so a model can score well overall while
# barely predicting rare types 3–7. Precision and recall per class show whether
# each forest type is actually learned.

report.metrics.summarize(
    metric=multiclass_metrics,
    data_source="both",
).frame(favorability=True)

# %%
# Use class weights
# =================
#
# ``class_weight="balanced"`` reweights the loss toward rare labels during
# training. SKD005 still fires because row proportions are unchanged — the
# check measures the dataset, not the loss.

from skore import compare

report_weighted = evaluate(
    HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=42,
    ),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# SKD005 remains present; only the data mix clears this check.

report_weighted.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
comparison_weights = compare({"default": report, "balanced_weights": report_weighted})
comparison_weights.metrics.summarize(
    metric=multiclass_metrics,
    data_source="both",
).frame(favorability=True)

# %%
# Collect more data for rare classes
# ==================================
#
# Append rows from the full Covertype table for forest types that still fall
# below the 10 % threshold until each rare class crosses the line.

X_augmented = X.reset_index(drop=True)
y_augmented = y.reset_index(drop=True)

while True:
    shares = y_augmented.value_counts(normalize=True)
    rare = shares[shares < 0.1]
    if rare.empty:
        break
    rarest_class = rare.idxmin()
    n_current = (y_augmented == rarest_class).sum()
    n_needed = int(np.ceil(0.1 * len(y_augmented) / 0.9)) - n_current
    n_needed = max(n_needed, 1)
    pool = X_full[y_full == rarest_class]
    n_needed = min(n_needed, len(pool))
    extra_idx = pool.sample(n=n_needed, random_state=42).index
    X_augmented = pd.concat(
        [X_augmented, X_full.loc[extra_idx].reset_index(drop=True)],
        ignore_index=True,
    )
    y_augmented = pd.concat(
        [y_augmented, y_full.loc[extra_idx].reset_index(drop=True)],
        ignore_index=True,
    )

y_augmented = pd.Series(y_augmented, name="class")

# %%
y_augmented.value_counts(normalize=True).round(4)

# %%
report_more_data = evaluate(
    classifier,
    X=X_augmented,
    y=y_augmented,
    splitter=TrainTestSplit(
        test_size=0.2,
        random_state=42,
        stratify=y_augmented,
    ),
)

# %%
# With enough rare-type rows added, SKD005 should be absent.

report_more_data.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
report_more_data.metrics.summarize(
    metric=multiclass_metrics,
    data_source="both",
).frame(favorability=True)

# %%
# Resample the dataset
# ====================
#
# Oversampling duplicates existing rare-class rows rather than adding new ones.
# Here, we oversample rare classes with replacement until every class matches the majority
# count in the 10,000-row subsample, then score on a larger stratified holdout
# taken only from rows that were never in that subsample to avoid train/test overlap.

from sklearn.utils import resample
from skore import EstimatorReport


def rebalance_by_oversampling(X, y, random_state):
    y = pd.Series(y).reset_index(drop=True)
    X = pd.DataFrame(X).reset_index(drop=True)
    target_n = y.value_counts().max()

    X_parts, y_parts = [], []
    for cls in y.unique():
        mask = y == cls
        X_cls, y_cls = resample(
            X.loc[mask],
            y.loc[mask],
            replace=True,
            n_samples=target_n,
            random_state=random_state,
        )
        X_parts.append(X_cls)
        y_parts.append(y_cls)

    X_out = pd.concat(X_parts, ignore_index=True)
    y_out = pd.concat(y_parts, ignore_index=True)
    return resample(
        X_out,
        y_out,
        replace=False,
        n_samples=len(y_out),
        random_state=random_state,
    )


X_train_oversampled, y_train_oversampled = rebalance_by_oversampling(
    X, y, random_state=42
)

y_train_oversampled.value_counts(normalize=True).round(4)

# %%
# Draw the test fold from ``X_holdout`` / ``y_holdout`` so it shares no rows with
# the oversampled training set.

X_test_large, _, y_test_large, _ = train_test_split(
    X_holdout,
    y_holdout,
    train_size=20_000,
    stratify=y_holdout,
    random_state=42,
)
y_test_large = pd.Series(y_test_large, name="class")

report_resampled = EstimatorReport(
    HistGradientBoostingClassifier(random_state=42),
    X_train=X_train_oversampled,
    y_train=y_train_oversampled,
    X_test=X_test_large,
    y_test=y_test_large,
)

# %%
# SKD005 looks at train and test together. The holdout still has the original distribution
# of classes, so the check can remain for some classes even though training labels are oversampled.

report_resampled.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Train scores can look strong because rare rows are repeated in the training
# set. Test metrics on the 20,000-row holdout show how the model behaves on the
# natural class mix.

report_resampled.metrics.summarize(
    metric=multiclass_metrics,
    data_source="both",
).frame(favorability=True)

# %%
# Conclusion
# ==========
#
# SKD005 is a reminder to look beyond overall accuracy when some classes are rare.
# A high accuracy score can hide near-zero recall on types that barely appear in
# the data. Per-class metrics make that gap visible, class weights can shift
# training toward rare labels, and oversampling or adding more rare-class rows
# can rebalance the dataset when SKD005 fires.
