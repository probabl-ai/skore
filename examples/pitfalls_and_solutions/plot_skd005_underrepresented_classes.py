"""
.. _example_skd005_underrepresented_classes:

SKD005 - Underrepresented classes
=================================

:ref:`SKD005 <skd005-underrepresented-classes>` flags a multiclass task when
one or more classes each represent less than 10 % of rows. Overall accuracy can
look acceptable while rare labels are barely learned. This notebook is about
how to work with that rarity once the check fires: we do not try to make SKD005
disappear by reshaping the class histogram.

What to do instead:

- report absolute counts as well as percentages,
- evaluate threshold-free / probabilistic metrics (such as log-loss) before
  per class precision and accuracy,
- collect more rare-class labels when possible, without treating a cleared
  check as success,
- correct for prevalence shift if acquisition oversamples rare types.

For binary rare-event tasks (threshold tuning, when ``class_weight`` is a risky
shortcut), see :ref:`skd004-high-class-imbalance` and:
https://probabl-ai.github.io/calibration-cost-sensitive-learning/content/notebooks/imbalanced_classification.html

We take a 10,000-row stratified subsample of Covertype so several forest types
fall below 10 %. The goal is to keep natural prevalence visible, judge the
multiclass model honestly, then see what extra rare-class rows can do.
"""

# %%
# Load the Covertype dataset
# ==========================
#
# The full Covertype task has seven forest types. A small stratified subsample
# keeps frequent classes well represented while types 3-7 drop below 10 % each.
# We keep the unused rows as a pool for the "more rare-class data" section later.

import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

df = fetch_covtype(as_frame=True).frame
y_full = df["Cover_Type"].astype(str)
X_full = df.drop(columns=["Cover_Type"])

X, X_pool, y, y_pool = train_test_split(
    X_full,
    y_full,
    train_size=10_000,
    stratify=y_full,
    random_state=42,
)
y = pd.Series(y, name="class")
y_pool = pd.Series(y_pool, name="class")

# %%
# Let us inspect the feature matrix with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
# Clicking the column in the target's `TableReport` brings
# a class histogram that shows that the classes are not evenly distributed.

TableReport(y)

# %%
# Let us also look at absolute counts: type 4 is under 1 % with only a
# few dozen rows in this subsample: even a good multiclass model has little to
# learn from there. Types such as 6 are also under the 10 % SKD005 bar, but with
# a few hundred rows they are less hopeless.

shares = y.value_counts(normalize=True).sort_index()
counts = y.value_counts().sort_index()
print(pd.DataFrame({"share": shares.round(4), "count": counts}))

# %%
# Trigger SKD005: default classifier on imbalanced classes
# ========================================================
#
# A default gradient boosting classifier does not change label counts. We ignore
# SKD008 to avoid correlated-feature warnings from Covertype's constant soil
# one-hots.

import skore
from sklearn.ensemble import HistGradientBoostingClassifier
from skore import TrainTestSplit

splitter = TrainTestSplit(test_size=0.2, random_state=42, stratify=y)
classifier = HistGradientBoostingClassifier(random_state=42)

report = skore.evaluate(
    classifier,
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# SKD005 correctly flags classes 3, 4, 5, 6, and 7 as under 10 % of rows.

report.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Accuracy hides rare-class failures
# ==================================
#
# Accuracy alone can look strong when frequent classes dominate the table.
# Let us however report it with per class precision and log-loss.
# We see that global accuracy hides rare-class failures, such as for types
# 4 and 5.

report.metrics.summarize(metric=["accuracy", "precision", "log_loss"]).frame(
    flat_index=False
)

# %%
# Let us also inspect the confusion matrix. We see that the model has a hard time
# predicting type 6, often confusing it with types 2 and 3 or when predicting type 7,
# confusing it with type 1.

report.metrics.confusion_matrix().plot()


# %%
# More rare-class training data
# =============================
#
# Extra labels on underrepresented types can help the multiclass model see them
# more often. Let us keep one fixed test fold with the natural mix, fit on the
# original train fold, then refit after adding rare-class rows from the pool
# (classes that were under 10 % in the subsample).

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rare_labels = shares[shares < 0.10].index
pool_rare = y_pool.isin(rare_labels)
X_rare_extra, _, y_rare_extra, _ = train_test_split(
    X_pool.loc[pool_rare],
    y_pool.loc[pool_rare],
    train_size=min(5_000, int(pool_rare.sum())),
    stratify=y_pool.loc[pool_rare],
    random_state=42,
)
X_train_more = pd.concat([X_train, X_rare_extra])
y_train_more = pd.concat([y_train, y_rare_extra])

print("Rare labels added from the pool:", list(rare_labels))
print("Extra rare-class rows added:", len(y_rare_extra))
print("\nBaseline train counts:")
print(y_train.value_counts().sort_index())
print("\nEnriched train counts:")
print(y_train_more.value_counts().sort_index())

# %%
# Let us now fit a model on the enriched train set and compare the results with the
# original model. We can observe that the model on the enriched train set has a better
# log-loss, accuracy and per-class precision on the common test set.
#
# The log-loss is the most importance metric to look at here, as it evaluates the model's
# predicted probabilities, which give more robust estimate of the model's quality.
# In contrast, accuracy and per-class precision are computed with hard class predictions,
# obtained from the argmax of the predicted probabilities, which can hide uncalibrated predictions.

model_less = HistGradientBoostingClassifier(random_state=42).fit(X_train, y_train)
report_less = skore.evaluate(model_less, X_test, y_test, splitter="prefit")

model_more = HistGradientBoostingClassifier(random_state=42).fit(
    X_train_more, y_train_more
)
report_more = skore.evaluate(model_more, X_test, y_test, splitter="prefit")

skore.compare(
    {
        "baseline_train": report_less,
        "more_rare_class_rows": report_more,
    }
).metrics.summarize(
    metric=["accuracy", "precision", "log_loss"],
    data_source="test",
).frame(flat_index=False)

# %%
# Enriching the train set with more rare-class rows also clears SKD005.

report_more.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Collecting more rare-class rows can improve rare-class precision and log-loss on
# a fixed natural-prevalence test set. That does not mean we should chase a
# cleared SKD005: if acquisition preferentially samples rare types, the training
# mix no longer matches the field, and production prevalence may stay low. We
# can correct for that shift before reading operating metrics. Clearing the
# check by reshaping the histogram is optional; better rare-class decisions
# under honest prevalence is the point.

# %%
# Conclusion
# ==========
#
# SKD005 is a multiclass rarity warning, not a request to rebalance at all
# costs. We prefer log-loss (and confusion matrices) over accuracy, we report
# absolute counts, and we can add rare-class labels when we can without treating
# a silent check as success.
#
# For binary rare-event threshold tuning and when
# ``class_weight`` is a risky shortcut, see
# :ref:`skd004-high-class-imbalance`.
