"""
.. _example_skd003_inconsistent_performance:

SKD003 - Inconsistent performance across splits
===============================================

:ref:`SKD003 <skd003-inconsistent-performance>` flags folds whose test metrics
diverge sharply from the median on a :class:`~skore.CrossValidationReport`.
With a proper splitter this is often a diagnostic check: the data have structure
(groups, time, or a corrupted batch) that shuffled CV would hide.

Realistic triggers:

- a contiguous bad batch of labels or features (mislabelled window, logging
  bug, schema mix-up),
- a much easier or harder group in one test fold under
  :class:`~sklearn.model_selection.GroupKFold`,
- temporal drift under :class:`~sklearn.model_selection.TimeSeriesSplit`
  (e.g. an easy class that becomes rarer),
- accidental fold imbalance from unshuffled
  :class:`~sklearn.model_selection.KFold` when prevalence varies along
  collection order (then shuffle or stratify if that will not appear in
  production).

When structure is real, shuffled CV overestimates performance. SKD003 under a
proper split is a good sign. The structure may not be fully fixable: once
understood, mute with :func:`~skore.configuration` and consider collecting more
data on the hard regime.

This notebook walks four beats: artificial corruption, a bad group in test,
distribution shift in the last time-series fold, then ignoring SKD003 once the
problem is understood. We focus on SKD003 throughout and pass
``ignore=["SKD008"]`` so Covertype's constant one-hot soil columns do not
drown the summary in correlated-feature noise.
"""

# %%
# Load Covertype (two classes)
# ============================
#
# Classes 1 and 2 dominate Covertype. We keep those rows and take a 2,000-row
# subsample. SKD003 needs CV, so we use ``splitter=5`` or an explicit five-fold
# splitter below.

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
# Inspect features with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
# The two classes are fairly balanced on this subsample.

TableReport(y_clean)

# %%
# Artificial corruption: a bad contiguous batch
# =============================================
#
# Randomly reassign labels on the first fifth of rows (mislabelled batch,
# logging bug, or merge mix-up). ``splitter=5`` is unshuffled
# :class:`~sklearn.model_selection.KFold`, so that block stays in fold #0 and
# SKD003 can flag it.

y_corrupted = y_clean.copy()
n_corrupt = len(y_corrupted) // 5
y_corrupted.iloc[:n_corrupt] = np.random.RandomState(0).choice(
    y_clean.unique(),
    size=n_corrupt,
)

# %%
# Default :func:`~skrub.tabular_pipeline` classifier for preprocessing.

from skrub import tabular_pipeline

model = tabular_pipeline("classifier")

# %%
# Trigger SKD003: cross-validate on corrupted labels
# ==================================================
#
# Corrupted rows land in split #0; later folds stay clean. Inspect per-split
# metrics with ``aggregate=None``, then checks.

import skore

report = skore.evaluate(model, X=X, y=y_corrupted, splitter=5, n_jobs=4)

report.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
# ``SKD003`` should flag split #0.

report.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Debug the flagged fold through its EstimatorReport
# ==================================================
#
# ``reports_`` holds one :class:`~skore.EstimatorReport` per split; open the
# flagged fold as usual.

outlier_report = report.reports_[0]
outlier_report.metrics.summarize(data_source="test").frame()

# %%
outlier_report.metrics.precision_recall().plot()

# %%
# If the cause was a fixable bad batch, clean labels clear the outlier fold.

report_clean = skore.evaluate(model, X=X, y=y_clean, splitter=5, n_jobs=4)
report_clean.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
report_clean.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Bad group in the test fold
# ==========================
#
# With group ids (site, patient, batch), a grouped splitter keeps each group on
# one side of every split. A much harder or easier group in one test fold
# triggers SKD003: expected, the splitter did its job. Shuffled CV would smear
# that group across folds, hide the gap, and overestimate performance.
#
# Below we invent batch ids, corrupt labels for batch ``0`` only, and evaluate
# with :class:`~sklearn.model_selection.GroupKFold`.

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

group_splits = list(GroupKFold(n_splits=5).split(X, y_batch, groups=groups))
report_grouped = skore.evaluate(model, X=X, y=y_batch, splitter=group_splits, n_jobs=4)
report_grouped.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
# ``SKD003`` should flag the fold that tests the corrupted batch.

report_grouped.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Distribution shift in the last time-series fold
# ===============================================
#
# Under :class:`~sklearn.model_selection.TimeSeriesSplit`, later windows can
# diverge from early training data (e.g. an easy class becomes rarer). Earlier
# folds look strong, the last fold drops, and SKD003 fires.
#
# Build a balanced task with :func:`~sklearn.datasets.make_classification`,
# treat row order as time, and replace labels only in the final test window
# with a rare positive class so that one late fold stands out. No shuffle.

from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

rng = np.random.RandomState(0)
n_time = 3_000
n_splits = 5
# Last test fold is the final ``n_time // (n_splits + 1)`` rows.
hard_start = n_time - (n_time // (n_splits + 1))

X_arr, y_arr = make_classification(
    n_samples=n_time,
    n_features=6,
    n_informative=2,
    n_redundant=0,
    weights=[0.5, 0.5],
    random_state=0,
)
X_time = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(6)])
y_time = pd.Series(y_arr, name="label")
y_time.iloc[hard_start:] = rng.choice([0, 1], size=n_time - hard_start, p=[0.96, 0.04])

time_splits = list(TimeSeriesSplit(n_splits=n_splits).split(X_time))
model_time = HistGradientBoostingClassifier(random_state=0)
report_time = skore.evaluate(
    model_time, X=X_time, y=y_time, splitter=time_splits, n_jobs=4
)
report_time.metrics.summarize(data_source="test").frame(aggregate=None)

# %%
# Split #4 should underperform once positives are rare in that window. SKD003
# here is expected from chronological CV, not a reason to reshuffle time.

report_time.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# When the problem is understood: mute SKD003
# ===========================================
#
# Groups and drift are data properties: investigate, collect more labels on the
# hard regime if needed, but folds may never look uniform. Once SKD003 is
# expected, mute it with :func:`~skore.configuration` (or
# ``ignore=["SKD003"]`` on one summarize call).

with skore.configuration(ignore_checks=["SKD003"]):
    muted = report_time.checks.summarize(fast_mode=True, ignore=["SKD008"])
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
# SKD003 is a reminder to inspect unstable cross-validation folds. With a
# proper splitter, firing often means the evaluation exposed a bad batch,
# a hard group, or temporal shift. Fix what you can (for example a corrupted
# label window). When the structure is intrinsic, keep the honest splitter,
# document the outlier regime, mute SKD003 via configuration, and collect more
# data on that regime if you need better coverage. Avoid shuffled CV as a way
# to make the check disappear when groups or time are real.
# See also :ref:`SKD013 <skd013-train-test-time-overlap>` for chronological
# train/test overlap on hold-out reports.
