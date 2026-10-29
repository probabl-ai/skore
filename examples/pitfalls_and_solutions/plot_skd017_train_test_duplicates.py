"""
.. _example_skd017_train_test_duplicates:

SKD017 - Duplicate rows across train and test
=============================================

This example shows what check :ref:`SKD017 <skd017-train-test-duplicates>`
catches and how to fix it.

The idea is simple. The only way the exact same row can end up in both the
training set and the test set after a random split is if that row appears more
than once in the original dataset. When this happens, the model is tested on
rows it has already seen during training, so the score looks better than it
really is. This is a form of *data leakage*.

Instead of comparing the train and test sets after the split, SKD017 looks at
the **full dataset before splitting** and counts how many rows are duplicates of
another row. If there are duplicates, splitting without removing them first is
likely to leak training rows into the test set.

We use the ``kddcup99`` dataset from scikit-learn: it describes network
connections and is famous for containing a lot of repeated rows. It is also
large, so we take a small random sample to keep the example fast.

Mitigations from the :ref:`automated_checks` user guide:

- de-duplicate the dataset before splitting into train and test,
- if the duplicates are legitimate, use a grouped splitter such as
  :class:`~sklearn.model_selection.GroupShuffleSplit` so identical rows stay on
  the same side of the split.
"""

# %%
# Load a small sample of the kddcup99 dataset
# ===========================================
#
# We load the dataset as a pandas DataFrame and keep only a small random sample
# (the full dataset has hundreds of thousands of rows). The target tells whether
# a connection is an attack (``1``) or normal traffic (``0``).

from sklearn.datasets import fetch_kddcup99

dataset = fetch_kddcup99(percent10=True, as_frame=True, random_state=0)
df = dataset.frame.sample(n=3000, random_state=0).reset_index(drop=True)

y = (df["labels"] != b"normal.").astype(int)
X = df.drop(columns=["labels"])

# %%
# Many rows are exact copies of another row. The percentage below is high: more
# than half of the sampled rows have at least one identical twin.

duplicate_share = X.duplicated(keep=False).mean()
print(f"{duplicate_share:.1%} of the rows are duplicates of another row.")

# %%
# :class:`~skrub.TableReport` gives a quick overview of the features.

from skrub import TableReport

TableReport(X)

# %%
# Trigger SKD017 - split without de-duplication
# =============================================
#
# We fit a tabular classifier with a plain random train/test split. Because the
# dataset is full of duplicates, the same rows land on both sides of the split.

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
# The test scores look almost perfect. That is suspicious: the model is being
# graded on rows it already memorized during training.

report.metrics.summarize(data_source="test").frame()

# %%
# Fix it - de-duplicate before splitting
# ======================================
#
# We drop the duplicate rows so that every row in the dataset is unique, then
# evaluate again on this cleaned data.

mask = ~df.duplicated()
df_unique = df[mask].reset_index(drop=True)

y_unique = (df_unique["labels"] != b"normal.").astype(int)
X_unique = df_unique.drop(columns=["labels"])

print(f"Rows before de-duplication: {len(df)}")
print(f"Rows after de-duplication:  {len(df_unique)}")

# %%
# Evaluate on the de-duplicated dataset.

report_unique = evaluate(
    tabular_pipeline("classifier"),
    X=X_unique,
    y=y_unique,
    splitter=TrainTestSplit(random_state=0),
)
report_unique

# %%
# SKD017 is gone: there are no more duplicate rows to leak across the split.

report_unique.checks.summarize(fast_mode=True)

# %%
# The test scores are now computed on rows the model has never seen, so they are
# a more honest estimate of how the model generalizes.

report_unique.metrics.summarize(data_source="test").frame()

# %%
# Conclusion
# ==========
#
# SKD017 flags duplicate rows in your dataset before they can leak across a
# train/test split. The fix is usually to remove the duplicates before
# splitting. If the repeated rows are meaningful and must be kept, split with a
# grouped strategy so that identical rows never end up on both sides at once.
