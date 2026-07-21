"""
.. _example_skd003_inconsistent_performance:

SKD003 — Inconsistent performance across splits
===============================================

This example demonstrates mitigations when check :ref:`SKD003 <skd003-inconsistent-performance>`
fires on a :class:`~skore.CrossValidationReport`. The check compares per-fold test scores and flags
splits whose metrics diverge sharply from the median — often a sign of a bad data batch, leakage, or
an unlucky fold draw.

Mitigations from the :ref:`automated_checks` user guide:

- use grouped cross-validation when observations share a group structure,
- investigate whether the outlier split has a different distribution,
- check for data leakage or temporal effects,
- increase the size of the dataset to improve stability.

We use a subsample of the Covertype classification task, restricted to its two most frequent forest
cover types, and corrupt labels in a contiguous block of rows so one fold becomes an outlier. The
goal is to restore homogeneous cross-validation scores once the root cause is addressed.
"""

# %%
# Load Covertype (two classes)
# ============================
#
# The Covertype task predicts forest cover type from cartographic variables. Classes 1 (Spruce/Fir)
# and 2 (Lodgepole Pine) make up about 85 % of the table. We keep only those rows and draw a
# 2,000-row subsample. Because SKD003 applies only to cross-validated reports, every evaluation
# below uses ``splitter=5`` (or an explicit CV splitter with five folds).

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
    random_state=0,
)

# %%
# Inspect the feature matrix with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
# The two retained classes are fairly balanced on this subsample.

TableReport(y_clean)

# %%
# Corrupt labels to simulate a bad production batch
# =================================================
#
# We copy the clean target and randomly reassign labels on the first fifth of rows. That mimics a
# real incident: a mislabelled annotation batch, a logging bug that swapped class codes for one
# window of production data, or a schema mix-up when merging sources. The corrupted block is large
# enough that one cross-validation fold will look like an outlier under SKD003, which is the
# situation we want to trigger and then debug.

y_corrupted = y_clean.copy()
n_corrupt = len(y_corrupted) // 5
y_corrupted.iloc[:n_corrupt] = np.random.RandomState(0).choice(
    y_clean.unique(),
    size=n_corrupt,
)

# %%
# A default :func:`~skrub.tabular_pipeline` classifier handles the preprocessing.

from skrub import tabular_pipeline

model = tabular_pipeline("classifier")

# %%
# Trigger SKD003 — cross-validate on corrupted labels
# ===================================================
#
# Five-fold cross-validation shuffles rows before splitting, so the corrupted prefix is spread
# across folds — but fold #0 still sees enough bad labels to drag its test scores away from the
# others. Inspect per-split metrics with ``aggregate=None``, then summarize checks.

from skore import evaluate

report = evaluate(model, X=X, y=y_corrupted, splitter=5, n_jobs=4)

report.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
# SKD003 should list split #0 as an outlier relative to the other folds. We ignore SKD008 here: its
# correlated-features check is unrelated to this walkthrough and can warn on Covertype's constant
# one-hot soil columns.

report.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Debug the flagged fold through its EstimatorReport
# ==================================================
#
# A :class:`~skore.CrossValidationReport` stores one :class:`~skore.EstimatorReport` per split in
# ``reports_``. Open the flagged fold to inspect its metrics and plots the same way you would for a
# single hold-out report.

outlier_report = report.reports_[0]
outlier_report.metrics.summarize(data_source="test").frame()

# %%
outlier_report.metrics.precision_recall().plot()

# %%
# Investigate and fix the outlier split
# =====================================
#
# In this example we know the root cause because we created it: the first fifth of labels was
# flipped. The fix is to swap back to ``y_clean`` and show that SKD003 clears once the batch is
# corrected.

report_clean = evaluate(model, X=X, y=y_clean, splitter=5, n_jobs=4)

report_clean.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
# What to check when the cause is unknown? SKD003 does not distinguish label errors from leakage or
# temporal drift. If one fold still looks wrong after auditing labels, review the feature pipeline
# for target leakage (e.g. statistics fit on the full table before splitting, duplicate rows across
# folds, features that encode future information) and, when rows are ordered in time, whether
# cross-validation respects chronology — see :ref:`SKD013 <skd013-train-test-time-overlap>`. With
# clean labels, per-fold scores should cluster and SKD003 should be absent.

report_clean.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Use grouped cross-validation
# ============================
#
# When rows share a group id (site, patient, acquisition batch), ignoring that structure can put
# related rows in both train and test, or concentrate whole groups in a few folds, which inflates
# fold-to-fold variance. Below we invent batch ids and corrupt an entire batch — a common production
# failure mode — then compare a plain shuffle split to :class:`~sklearn.model_selection.GroupKFold`.

from sklearn.model_selection import GroupKFold

n_batches = 10
groups = pd.Series(
    np.minimum(np.arange(len(X)) // (len(X) // n_batches), n_batches - 1),
    index=X.index,
    name="batch_id",
)

y_batch = y_clean.copy()
bad_batch = groups == 0
y_batch.loc[bad_batch] = np.random.RandomState(1).choice(
    y_clean.unique(),
    size=int(bad_batch.sum()),
)

# %%
# Ignoring groups: related rows from the bad batch can land in several folds.

report_ignore_groups = evaluate(model, X=X, y=y_batch, splitter=5, n_jobs=4)
report_ignore_groups.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
report_ignore_groups.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Respecting groups: each batch stays on one side of every split, so a bad batch shows up as a clear
# outlier fold instead of leaking across folds. When observations share a group structure (sites,
# patients, acquisition batches), prefer :class:`~sklearn.model_selection.GroupKFold` (or another
# grouped splitter): it makes problematic batches visible under SKD003 instead of smearing them
# across folds and masking the issue.

group_splits = list(GroupKFold(n_splits=5).split(X, y_batch, groups=groups))
report_grouped = evaluate(model, X=X, y=y_batch, splitter=group_splits, n_jobs=4)
report_grouped.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
report_grouped.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Increase dataset size with clean rows
# =====================================
#
# Appending 4,000 clean rows from the full two-class table dilutes the corrupted share. Per-fold
# metrics become more homogeneous even though the bad batch is still mislabelled, because more data
# can mask, but not fix, label errors.

X_more, _, y_more, _ = train_test_split(
    X_full,
    y_full,
    train_size=4_000,
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
    n_jobs=4,
)

report_more_data.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
report_more_data.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Compare clean vs corrupted labels
# =================================
#
# :func:`~skore.compare` contrasts mean test metrics between the corrupted and clean reports on the
# same model and split count.

from skore import compare

comparison = compare({"clean_labels": report_clean, "corrupted_labels": report})
comparison.metrics.summarize(data_source="test").frame(aggregate="mean")

# %%
# Conclusion
# ==========
#
# SKD003 highlights unstable cross-validation — often a data or splitting issue rather than a
# hyperparameter problem. In this walkthrough, restoring clean labels removed the outlier fold:
# fixing the root cause is the main fix. Grouped cross-validation and more data are supporting
# levers for stable evaluation. Use grouped splits when rows share a site or batch id so bad batches
# stay visible under SKD003; add data only after the table is trustworthy, since extra clean rows
# can mask leftover label errors without repairing them.
