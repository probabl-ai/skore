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
# We use the breast cancer dataset shipped with scikit-learn: 569 tumors
# described by numerical measurements, where the target marks whether the tumor
# is malignant or benign. It has no duplicate rows on its own, so to demonstrate
# the pitfall we duplicate a random subset of the rows (sampled without
# replacement), mimicking a dataset where some records were recorded twice.

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer(as_frame=True)
df = dataset.data.assign(target=dataset.target)

rng = np.random.default_rng(0)
duplicated_rows = rng.choice(df.index, size=len(df) // 2, replace=False)
df = (
    pd.concat([df, df.loc[duplicated_rows]], ignore_index=True)
    .sample(frac=1.0, random_state=0)
    .reset_index(drop=True)
)

y = df["target"]
X = df.drop(columns="target")

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
# Because of the duplication, these test scores must be too optimistic. Below,
# we will see what a more honest evaluation looks like.

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
# If the duplicates are legitimate and you want to keep every observation, the
# splitter must keep all copies of a row on the same side of the split. Real
# datasets do not come with a group identifier, so we write a custom splitter
# that builds the groups itself by hashing each row: identical rows get the same
# hash, hence the same group. It then delegates to
# :class:`~sklearn.model_selection.GroupShuffleSplit`.

from sklearn.model_selection import GroupShuffleSplit


class DuplicateGroupShuffleSplit(GroupShuffleSplit):
    """Shuffle split that keeps identical rows on the same side of the split."""

    def split(self, X, y=None, groups=None):
        groups = pd.util.hash_pandas_object(X, index=False)
        return super().split(X, y, groups)


grouped_split = DuplicateGroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
report_grouped = evaluate(
    tabular_pipeline("classifier"),
    X=X,
    y=y,
    splitter=grouped_split.split(X, y),
)
report_grouped

# %%
# SKD017 still flags the duplicates, since they really are in the dataset.

report_grouped.checks.summarize()

# %%
# However, the test scores are back to an honest level because no copy crosses
# the split.

report_grouped.metrics.summarize(data_source="test").frame()

# %%
# Fix 3 - de-duplicate and use sample weights
# ===========================================
#
# Another option for legitimate duplicates is to collapse identical rows into a
# single row and weight it by how many times it occurred. The model sees each
# distinct row once, with a weight that preserves its importance, and no row can
# leak across the split.

weighted = (
    X.assign(target=y.to_numpy()).value_counts(dropna=False).reset_index(name="weight")
)
X_weighted = weighted[X.columns]
y_weighted = weighted["target"]
sample_weight = weighted["weight"].to_numpy()

print(f"Distinct rows: {len(X_weighted)}  (total weight: {sample_weight.sum()})")

# %%
# We split the distinct rows, fit with the per-row weights, and evaluate the
# already fitted model on the held-out rows.

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

report_weighted = evaluate(model, X_test, y_test, splitter="prefit")
report_weighted.metrics.summarize(data_source="test").frame()

# %%
# Conclusion
# ==========
#
# SKD017 flags duplicate rows before they can leak across a train/test split and
# inflate your scores. If the duplicates are not legitimate, remove them before
# splitting. If they are legitimate, either group identical rows so they never
# split apart, or de-duplicate and carry their count as a sample weight.
