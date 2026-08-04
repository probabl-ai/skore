"""
.. _example_skd005_underrepresented_classes:

SKD005 - Underrepresented classes
=================================

:ref:`SKD005 <skd005-underrepresented-classes>` flags a multiclass task when
one or more classes each represent less than 10 % of rows. Overall accuracy can
look acceptable while rare labels are barely learned. This notebook is about
how to work with that rarity once the check fires: we do not try to make SKD005
disappear by reshaping the class histogram.

What to do instead (see also :ref:`automated_checks`):

- report absolute counts as well as percentages,
- evaluate threshold-free / probabilistic metrics (especially log-loss) before
  trusting per-class precision / recall,
- treat per-class precision / recall as symptoms of the default multiclass
  decision rule (argmax over class probabilities),
- collect more rare-class labels when possible, without treating a cleared
  check as success,
- correct for prevalence shift if acquisition oversamples rare types.

For binary rare-event tasks (threshold tuning, when ``class_weight`` is a risky
shortcut), see :ref:`example_skd004_high_class_imbalance` and:
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
# We keep the unused rows as a pool for the "more rare-class data" beat later.

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

print("Relative frequencies:")
print(y.value_counts(normalize=True).sort_index().round(4))
print("\nAbsolute counts:")
print(y.value_counts().sort_index())

# %%
# Let us inspect the feature matrix with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
# The class histogram shows that the classes are not evenly distributed and
# types 3-7 will trigger SKD005.

TableReport(y)

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
# Accuracy alone can look strong when frequent classes dominate the table. We
# report it together with log-loss. We do not use F1: it averages precision and
# recall and is a poor default under imbalance. Precision and recall below are
# thresholded symptoms at the model's default multiclass decision rule (argmax
# of predicted probabilities), not the main quality scoreboard.

report.metrics.summarize(
    metric=["accuracy", "precision", "recall", "log_loss"],
    data_source="both",
).frame()

# %%
report.metrics.confusion_matrix().plot()

# %%
print("Log-loss (test):", report.metrics.log_loss(data_source="test"))

# %%
# Absolute counts matter as much as percentages. Type 4 is under 1 % with only a
# few dozen rows in this subsample: even a good multiclass model has little to
# learn from there. Types such as 6 are also under the 10 % SKD005 bar, but with
# a few hundred rows they are less hopeless.

shares = y.value_counts(normalize=True).sort_index()
counts = y.value_counts().sort_index()
print(pd.DataFrame({"share": shares.round(4), "count": counts}))

# %%
# More rare-class training data
# =============================
#
# Extra labels on underrepresented types can help the multiclass model see them
# more often. Let us keep one fixed test fold with the natural mix, fit on the
# original train fold, then refit after adding rare-class rows from the pool
# (classes that were under 10 % in the subsample). We use
# :func:`~skore.evaluate` with ``splitter="prefit"`` so both models are scored
# on the same test rows.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rare_labels = shares[shares < 0.10].index
pool_rare = y_pool.isin(rare_labels)
# Cap how many extra rare rows we add so the gallery stays fast; the point is
# the direction of the effect, not using the entire Covertype remainder.
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
print("Train size (baseline):", len(y_train))
print("Train size (more rare-class rows):", len(y_train_more))
print("\nBaseline train counts:")
print(y_train.value_counts().sort_index())
print("\nEnriched train counts:")
print(y_train_more.value_counts().sort_index())

# %%
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
    metric=["accuracy", "precision", "recall", "log_loss"],
    data_source="test",
).frame()

# %%
# Collecting more rare-class rows can improve rare-class recall and log-loss on
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
# costs. We prefer log-loss (and confusion matrices) over F1, we report
# absolute counts, and we can add rare-class labels when we can without treating
# a silent check as success. For binary rare-event threshold tuning and when
# ``class_weight`` is a risky shortcut, see
# :ref:`example_skd004_high_class_imbalance`.
