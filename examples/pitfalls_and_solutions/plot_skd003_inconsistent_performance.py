"""
.. _example_skd003_inconsistent_performance:

SKD003 — Inconsistent performance across splits
===============================================

This example demonstrates mitigations when check
:ref:`SKD003 <skd003-inconsistent-performance>` fires on a
:class:`~skore.CrossValidationReport`. The check compares per-fold test scores
and flags splits whose metrics diverge sharply from the median — often a sign
of a bad data batch, leakage, or an unlucky fold draw.

Mitigations from the :ref:`automated_checks` user guide:

- use stratified or grouped cross-validation,
- investigate whether the outlier split has a different distribution,
- check for data leakage or temporal effects,
- increase the size of the dataset to improve stability.

We use a stratified subsample of the Covertype classification task, restricted
to its two most frequent forest cover types, and corrupt labels in the first
20 % of rows so fold #0 becomes an outlier. The goal is to restore homogeneous
cross-validation scores once the root cause is addressed.
"""

# %%
# Load Covertype (two classes) and corrupt labels
# ===============================================
#
# The Covertype task predicts forest cover type from cartographic variables.
# Classes 1 (Spruce/Fir) and 2 (Lodgepole Pine) make up about 85 % of the
# table. We keep only those rows, draw a 2,000-row stratified subsample, then
# flip labels on the first fifth to simulate a mislabelled production batch.
# Because SKD003 applies only to cross-validated reports, every evaluation below
# uses ``splitter=5``.
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

covtype = fetch_covtype(as_frame=True).frame
top_two = covtype["Cover_Type"].isin([1, 2])
y_full = covtype.loc[top_two, "Cover_Type"].astype(str)
X_full = covtype.loc[top_two].drop(columns=["Cover_Type"])

X, _, y_clean, _ = train_test_split(
    X_full,
    y_full,
    train_size=2_000,
    stratify=y_full,
    random_state=0,
)

# %%
# Inspect the feature matrix with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
# The two retained classes are fairly balanced. Stratified splitting keeps both
# visible in every fold so per-fold accuracy is meaningful.

TableReport(y_clean)

# %%
y_corrupted = y_clean.copy()
n_corrupt = len(y_corrupted) // 5
y_corrupted.iloc[:n_corrupt] = np.random.RandomState(0).choice(
    y_clean.unique(),
    size=n_corrupt,
)

# %%
# A default :func:`~skrub.tabular_pipeline` classifier handles the numeric
# inputs without extra preprocessing.

from skrub import tabular_pipeline

model = tabular_pipeline("classifier")

# %%
# Trigger SKD003 — cross-validate on corrupted labels
# ===================================================
#
# Five-fold cross-validation shuffles rows before splitting, so the corrupted
# prefix is spread across folds — but fold #0 still sees enough bad labels to
# drag its test scores away from the others. Inspect per-split metrics with
# ``aggregate=None``, then summarize checks.

from skore import evaluate

report = evaluate(model, X=X, y=y_corrupted, splitter=5, n_jobs=-1)

report.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
# SKD003 should list split #0 as an outlier relative to the other folds.
# We ignore SKD008 here: its correlated-features check is unrelated to this
# walkthrough and can warn on Covertype's constant one-hot soil columns.
report.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Investigate and fix the outlier split
# =====================================
#
# In this example we know the root cause because we created it. We flipped labels
# on the first fifth of rows to mimic a mislabelled production batch. The fix is
# to simply swap back to ``y_clean`` to show that SKD003 clears once the batch is
# fixed.

report_clean = evaluate(model, X=X, y=y_clean, splitter=5, n_jobs=-1)

report_clean.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
# What to check when the cause is unknown? SKD003 does not distinguish
# label errors from leakage or temporal drift. If one fold still looks wrong
# after auditing labels, review the feature pipeline for target leakage (e.g.
# statistics fit on the full table before splitting, duplicate rows across
# folds, features that encode future information) and, when rows are ordered in
# time, whether cross-validation respects chronology — see
# :ref:`SKD013 <skd013-train-test-time-overlap>`.
# %%
# With clean labels, per-fold scores should cluster and SKD003 should be absent.

report_clean.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Use stratified cross-validation
# ===============================
#
# :class:`~sklearn.model_selection.StratifiedKFold` keeps class proportions
# even in every fold. That improves split design for targets but does not remove
# corrupted labels — SKD003 may still fire on ``y_corrupted``.

from sklearn.model_selection import StratifiedKFold

report_stratified = evaluate(
    model,
    X=X,
    y=y_corrupted,
    splitter=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=0,
    ),
    n_jobs=-1,
)

report_stratified.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
report_stratified.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Increase dataset size with clean rows
# =====================================
#
# Appending 4,000 clean rows from the full two-class table dilutes the corrupted
# share. Per-fold metrics become more homogeneous even though the bad batch is
# still mislabelled — a reminder that more data can mask, but not fix, label
# errors.

X_more, _, y_more, _ = train_test_split(
    X_full,
    y_full,
    train_size=4_000,
    stratify=y_full,
    random_state=1,
)

X_augmented = pd.concat(
    [X.reset_index(drop=True), X_more.reset_index(drop=True)],
    ignore_index=True,
)
y_augmented = pd.concat(
    [y_corrupted.reset_index(drop=True), y_more.reset_index(drop=True)],
    ignore_index=True,
)

report_more_data = evaluate(
    model,
    X=X_augmented,
    y=y_augmented,
    splitter=5,
    n_jobs=-1,
)

report_more_data.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
report_more_data.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Compare clean vs corrupted labels
# =================================
#
# :func:`~skore.compare` contrasts mean test metrics between the corrupted and
# clean reports on the same model and split count.

from skore import compare

comparison = compare({"clean_labels": report_clean, "corrupted_labels": report})
comparison.metrics.summarize(data_source="test").frame(aggregate="mean")

# %%
# Conclusion
# ==========
#
# SKD003 highlights unstable cross-validation — often a data or splitting issue
# rather than a hyperparameter problem. In this walkthrough, restoring clean
# labels removed the outlier fold; stratification and more data are supporting
# levers when the underlying table is sound.
