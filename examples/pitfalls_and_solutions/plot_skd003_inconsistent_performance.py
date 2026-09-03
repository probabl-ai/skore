"""
.. _example_skd003_inconsistent_performance:

SKD003 - Inconsistent performance across splits
===============================================

:ref:`SKD003 <skd003-inconsistent-performance>` flags folds whose test metrics diverge
sharply from the median during a cross-validation evaluation. With a proper splitter
this is often a diagnostic check: the data have structure (groups, time, or a corrupted
batch) that shuffled cross-validation would hide.

Realistic triggers:

- a contiguous bad batch of labels or features (mislabelled window, logging
  bug, schema mix-up),
- a much easier or harder group in one test fold under
  :class:`~sklearn.model_selection.GroupKFold`,
- temporal drift under :class:`~sklearn.model_selection.TimeSeriesSplit`
  (e.g. more ill patients start showing up),
- accidental fold imbalance from unshuffled
  :class:`~sklearn.model_selection.KFold` when prevalence varies along
  collection order (then shuffle or stratify if that will not appear in
  production).

When structure is real, shuffled cross-validation overestimates performance. SKD003
under a proper split is a good sign. The structure may not be fully fixable: once
understood, mute with :func:`~skore.configuration` and consider collecting more data on
the hard regime.

This notebook walks four beats: artificial corruption, a bad group in test, distribution
shift in the last time-series fold, then ignoring SKD003 once the problem is understood.
"""

# %%
# Load Breast Cancer (two classes)
# ================================
#
# Let us use the Breast Cancer dataset to show how SKD003 can detect a batch of
# corrupted labels.

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target


# %%
# We can inspect the features and target with :class:`~skrub.TableReport`, and
# notice that it is a well curated dataset with two balanced classes (they
# appear in roughly equal proportions).

from skrub import TableReport

TableReport(X)

# %%

TableReport(y)

# %%
# Artificial corruption: a bad contiguous batch
# =============================================
#
# Let us permute the labels on the first fifth of the dataset to break any
# association between `X` and `y`. Later, we will use unshuffled 5-fold
# cross-validation, so all corrupted rows will land in the same fold. We expect
# the score of this fold to be low due to this corruption. While we are
# creating this defect artificially, this is a scenario that can happen in
# practice due to e.g. a logging bug, a broken sensor, or a merge mix-up.

y_corrupted, n_corrupt = y.copy(), len(y) // 5
rng = np.random.default_rng(seed=0)
y_corrupted.iloc[:n_corrupt] = rng.permutation(y_corrupted.iloc[:n_corrupt])


# %%
# We will use a default :func:`~skrub.tabular_pipeline` classifier for preprocessing, and
# a gradient boosting model for prediction.

from sklearn.linear_model import LogisticRegression
from skrub import tabular_pipeline

model = tabular_pipeline(LogisticRegression())
model

# %%
# Trigger SKD003: cross-validate on corrupted labels
# ==================================================
#
# We will now evaluate the model on the corrupted labels using unshuffled
# 5-fold cross-validation and look at non aggregated metrics values, to notice
# the discrepancy on the first fold.

import skore

report = skore.evaluate(model, X=X, y=y_corrupted, pos_label=1, splitter=5)

report.metrics.summarize(data_source="test").frame(aggregate=None, flat_index=False)

# %%
# Looking at the checks results, ``SKD003`` correctly flags split #0.

report.checks.summarize(fast_mode=True)

# %%
# Visualize the variance across splits
# ====================================
#
# The ROC display of a cross-validation report overlays the splits and summarizes them
# with a mean AUC and its standard deviation. It is built to show how much a model moves
# from one split to the next, rather than to identify a given split. Here the curves are
# widely spread and one of them sinks towards the chance level: this is the instability
# that ``SKD003`` reported.

_ = report.metrics.roc().plot()

# %%
# If the cause was a fixable bad batch, clean labels clear the outlier split.

report_clean = skore.evaluate(model, X=X, y=y, pos_label=1, splitter=5)
report_clean.metrics.summarize(data_source="test").frame(
    aggregate=None, flat_index=False
)

# %%
report_clean.checks.summarize(fast_mode=True)

# %%
# Bad group in the test fold
# ==========================
#
# When observations carry a group identifier, e.g. the medical center where
# patient data were collected, a grouped splitter keeps each group on one side of
# every split. If one group is much harder or easier to predict, the fold that tests
# it will look like an outlier and ``SKD003`` will fire. That is expected: the splitter
# did its job. Shuffled cross-validation would smear the difficult group across folds,
# hiding the gap and overestimating performance.

import skrub
from sklearn.model_selection import GroupKFold

n_batches = 10
batch_id = np.minimum(np.arange(len(X)) // (len(X) // n_batches), n_batches - 1)

y_batch = y.copy()
bad_batch = batch_id == 0
rng_batch = np.random.default_rng(seed=1)
y_batch.iloc[bad_batch] = rng_batch.choice(y.unique(), size=int(bad_batch.sum()))

df_batch = X.assign(batch_id=batch_id, target=y_batch)

# %%
# :class:`~sklearn.model_selection.GroupKFold` needs the group vector at split time,
# so we use a skrub :class:`~skrub.DataOp` to attach it to the data.
# :meth:`~skrub.DataOp.skb.mark_as_X` accepts a ``cv`` argument and
# ``split_kwargs`` for group ids. The resulting learner carries its own
# cross-validation scheme, so :func:`~skore.evaluate` needs no ``splitter``.

data = skrub.var("data", df_batch)
groups = data["batch_id"]
X_op = data.drop(columns=["batch_id", "target"]).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)
y_op = data["target"].skb.mark_as_y()
learner = X_op.skb.apply(model, y=y_op).skb.make_learner()

# %%
report_grouped = skore.evaluate(learner, data={"data": df_batch}, pos_label=1)
report_grouped.metrics.summarize(data_source="test").frame(
    aggregate=None, flat_index=False
)

# %%
# Looking at the checks results, ``SKD003`` correctly flags split #0.

report_grouped.checks.summarize(fast_mode=True)

# %%
# Distribution shift in the last time-series fold
# ===============================================
#
# Under :class:`~sklearn.model_selection.TimeSeriesSplit`, later windows can diverge
# from the training data. Consider a medical setting where diagnoses are collected over
# time: a sudden influx of ill patients near the end of the study shifts the class
# distribution. Earlier folds look strong because the model was trained on a balanced
# population, but the last fold drops as the positive class becomes rare. SKD003 fires,
# which is expected from chronological cross-validation and not a reason to reshuffle
# time.
#
# We reuse the same breast-cancer dataset, attach a fake timestamp, and reduce the
# prevalence of the positive class in the last test window.

from sklearn.model_selection import TimeSeriesSplit

n_splits_time = 5
last_test_start = len(X) - (len(X) // (n_splits_time + 1))

y_time = y.copy()
rng_time = np.random.default_rng(seed=2)
y_time.iloc[last_test_start:] = rng_time.choice(
    [0, 1], size=len(X) - last_test_start, p=[0.95, 0.05]
)

timestamps = pd.date_range("2020-01-01", periods=len(X), freq="D")
df_time = X.assign(timestamp=timestamps, target=y_time)

# %%
# As for the grouped section, we use a :class:`~skrub.DataOp` to declare the
# time-series split directly on the data.

data_time = skrub.var("data_time", df_time)
X_time_op = data_time.drop(columns=["timestamp", "target"]).skb.mark_as_X(
    cv=TimeSeriesSplit(n_splits=n_splits_time),
)
y_time_op = data_time["target"].skb.mark_as_y()
learner_time = X_time_op.skb.apply(model, y=y_time_op).skb.make_learner()

# %%
report_time = skore.evaluate(learner_time, data={"data_time": df_time}, pos_label=1)
report_time.metrics.summarize(data_source="test").frame(
    aggregate=None, flat_index=False
)

# %%
# Split #4 underperforms because the positive class is now rare in that window.
# SKD003 here is expected from chronological cross-validation, not a reason to
# reshuffle time.

report_time.checks.summarize(fast_mode=True)

# %%
# When the problem is understood: mute SKD003
# ===========================================
#
# Groups and drift are data properties: investigate, collect more labels on the
# hard regime if needed, but folds may never look uniform. Once SKD003 is
# expected, mute it with :func:`~skore.configuration` (or
# ``ignore=["SKD003"]`` on one summarize call).

with skore.configuration(ignore_checks=["SKD003"]):
    muted = report_time.checks.summarize(fast_mode=True)
muted

# %%
# Side note: if there is no group or time structure, but an unshuffled
# :class:`~sklearn.model_selection.KFold` still creates imbalanced folds because
# class prevalence varies along the collection order (for example a sensor
# failed for part of the dump), shuffling or stratifying is appropriate *when
# you are sure that irregularity is accidental and will not appear in
# production*. That case is the exception where changing the splitter to
# smooth folds is the right fix; do not use it to hide real groups or time
# drift.

# %%
# Conclusion
# ==========
#
# SKD003 is a reminder to inspect unstable cross-validation folds. With a proper
# splitter, firing often means the evaluation exposed a bad batch, a hard group, or
# temporal shift. Fix what you can (for example a corrupted label window). When the
# structure is intrinsic, keep the honest splitter, document the outlier regime, mute
# SKD003 via configuration, and collect more data on that regime if you need better
# coverage. Avoid shuffled cross-validation as a way to make the check disappear when
# groups or time are real. See also :ref:`SKD013 <skd013-train-test-time-overlap>` for
# chronological train/test overlap on hold-out reports.
