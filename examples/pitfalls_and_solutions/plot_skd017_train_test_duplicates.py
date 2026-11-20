"""
.. _example_skd017_train_test_duplicates:

SKD017 - Duplicate rows across train and test
=============================================

This example shows what check :ref:`SKD017 <skd017-train-test-duplicates>`
catches and how to fix it.

When a row appears more than once in the dataset, a random train/test split can
send identical copies to both sides. The model is then graded on rows it already
saw during training, so the test score looks better than it really is: this is a
form of *data leakage*. SKD017 detects it by counting, in the full dataset before
splitting, how many rows are exact duplicates of another row.

Mitigations from the :ref:`automated_checks` user guide, shown below:

- if the duplicates are **not legitimate**, de-duplicate before splitting;
- if they **are legitimate**, either group identical rows so they stay on the
  same side of the split, or de-duplicate and use sample weights to keep track
  of the repeated rows.
"""

# %%
# Build a dataset with duplicate rows
# ===================================
#
# We use the German credit dataset: 1000 loan applicants described by financial
# and personal attributes, where the target marks whether the applicant is a
# good credit risk (``1``) or a bad one (``0``). It has no duplicate rows on its
# own, so to demonstrate the pitfall we duplicate a random half of the rows,
# mimicking a dataset where the same records were recorded twice.

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

dataset = fetch_openml("credit-g", version=1, as_frame=True)
df = dataset.frame.copy()
df["target"] = (df["class"] == "good").astype(int)
df = df.drop(columns=["class"])

# Remember each original row so we can group duplicates together later on.
df["row_id"] = df.index

rng = np.random.default_rng(0)
duplicated_rows = rng.choice(df.index, size=len(df), replace=True)
df = (
    pd.concat([df, df.loc[duplicated_rows]], ignore_index=True)
    .sample(frac=1.0, random_state=0)
    .reset_index(drop=True)
)

groups = df["row_id"]
y = df["target"]
X = df.drop(columns=["target", "row_id"])

duplicate_share = X.duplicated(keep=False).mean()
print(f"{duplicate_share:.1%} of the rows are duplicates of another row.")

# %%
# A plain shuffle split sends identical rows to both sides. We can see it
# directly: split the row indices and count how many rows in the test set have
# an exact twin among the training rows.

from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(X.index, test_size=0.25, random_state=0)
train_rows = X.loc[train_idx]
leaked = X.loc[test_idx].apply(tuple, axis=1).isin(train_rows.apply(tuple, axis=1))
print(f"{leaked.mean():.1%} of the test rows also appear in the training set.")

# %%
# Trigger SKD017 - split without de-duplication
# =============================================
#
# We fit a tabular classifier with a plain random train/test split. Because of
# the duplicates, the same rows land on both sides of the split.

from skore import TrainTestSplit, evaluate
from skrub import tabular_pipeline

report = evaluate(
    tabular_pipeline("classifier"),
    X=X,
    y=y,
    splitter=TrainTestSplit(random_state=0),
)
report

# %%
# SKD017 reports how many rows are duplicates of another row in the full
# dataset.

report.checks.summarize()

# %%
# The test scores look suspiciously high: the model is being graded on rows it
# already memorized during training.

report.metrics.summarize(data_source="test").frame()

# %%
# Fix 1 - de-duplicate before splitting
# =====================================
#
# If the duplicates are not legitimate, drop them so every row is unique, then
# evaluate again. This is the right fix when repeated rows are an artifact of
# data collection rather than something the real world produces.

mask = ~X.duplicated()
X_unique = X[mask].reset_index(drop=True)
y_unique = y[mask].reset_index(drop=True)

print(f"Rows before de-duplication: {len(X)}")
print(f"Rows after de-duplication:  {len(X_unique)}")

report_unique = evaluate(
    tabular_pipeline("classifier"),
    X=X_unique,
    y=y_unique,
    splitter=TrainTestSplit(random_state=0),
)
report_unique

# %%
# SKD017 is gone: there are no more duplicate rows to leak across the split.

report_unique.checks.summarize()

# %%
# The test scores drop to a more honest level, computed on rows the model has
# never seen.

report_unique.metrics.summarize(data_source="test").frame()

# %%
# Fix 2 - keep duplicates but group them
# ======================================
#
# If the duplicates are legitimate and you want to keep every observation, use a
# grouped splitter such as :class:`~sklearn.model_selection.GroupShuffleSplit`.
# Passing the original row identity as the group keeps all copies of a row on the
# same side of the split, so none can leak. We pass the resulting single split to
# :func:`~skore.evaluate` directly.

from sklearn.model_selection import GroupShuffleSplit

grouped_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
report_grouped = evaluate(
    tabular_pipeline("classifier"),
    X=X,
    y=y,
    splitter=grouped_split.split(X, y, groups),
)
report_grouped

# %%
# SKD017 still flags the duplicates (they really are in the dataset), but the
# test scores are back to the honest level because no copy crosses the split.

report_grouped.metrics.summarize(data_source="test").frame()

# %%
# Fix 3 - de-duplicate and use sample weights
# ===========================================
#
# Another option for legitimate duplicates is to collapse identical rows into a
# single row and weight it by how many times it occurred. The model sees each
# distinct row once, with a weight that preserves its importance, and no row can
# leak across the split.

feature_columns = list(X.columns)
counts = (
    X.assign(target=y.to_numpy())
    .groupby(feature_columns + ["target"], dropna=False)
    .size()
    .reset_index(name="weight")
)
X_weighted = counts[feature_columns]
y_weighted = counts["target"]
sample_weight = counts["weight"].to_numpy()

print(f"Distinct rows: {len(X_weighted)}  (total weight: {sample_weight.sum()})")

# %%
# We split the distinct rows, fit with the per-row weights, and build an
# :class:`~skore.EstimatorReport` from the fitted model.

from skore import EstimatorReport

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_weighted,
    y_weighted,
    sample_weight,
    test_size=0.25,
    random_state=0,
    stratify=y_weighted,
)

model = tabular_pipeline("classifier")
model.fit(X_train, y_train, histgradientboostingclassifier__sample_weight=w_train)

report_weighted = EstimatorReport(model, X_test=X_test, y_test=y_test)
report_weighted.metrics.summarize(data_source="test").frame()

# %%
# Conclusion
# ==========
#
# SKD017 flags duplicate rows before they can leak across a train/test split and
# inflate your scores. If the duplicates are not legitimate, remove them before
# splitting. If they are legitimate, either group identical rows so they never
# split apart, or de-duplicate and carry their count as a sample weight.
