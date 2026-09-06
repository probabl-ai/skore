"""
.. _example_skd013_train_test_time_overlap:

SKD013 - Train-test overlap in time series
==========================================

This example demonstrates mitigations when check
:ref:`SKD013 <skd013-train-test-time-overlap>` fires on temporal data. The
check compares datetime columns in train and test folds and flags overlap when
the latest training timestamp is not strictly before the earliest test
timestamp.

Mitigations from the :ref:`automated_checks` user guide:

- use a time-based splitter such as
  :class:`~sklearn.model_selection.TimeSeriesSplit` or similar.

We use the employee salaries dataset ordered by hire date and predict whether
salary exceeds the median. The goal is to evaluate on future hires only so
test scores reflect forward-looking performance.
"""

# %%
# Load the employee salaries dataset
# ==================================
#
# Rows are sorted by ``date_first_hired``. We expose hire dates as a pandas
# ``timestamp`` column because SKD013 requires a datetime dtype. Shuffling
# before a hold-out split mixes later hires into training; that is the pattern
# SKD013 is designed to catch. Use full
# :meth:`~skore.EstimatorReport.checks.summarize` on the trigger;
# ``fast_mode=True`` on fix cells.

import pandas as pd
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
df = dataset.X.copy()
df["current_annual_salary"] = dataset.y
df["timestamp"] = pd.to_datetime(df["date_first_hired"])
df = df.sort_values("timestamp").reset_index(drop=True)

y = (df["current_annual_salary"] > df["current_annual_salary"].median()).astype(int)
X = df.drop(columns=["current_annual_salary"])

# %%
# :class:`~skrub.TableReport` confirms chronological ordering and mixed HR
# features.

from skrub import TableReport

TableReport(X)

# %%
# The binary target marks above-median earners, a classification view of the
# salary column.

TableReport(y.to_frame(name="high_earner"))

# %%
# Trigger SKD013 - shuffled train/test split
# ==========================================
#
# :class:`~skore.TrainTestSplit` with ``shuffle=True`` randomizes row order
# before cutting folds, so future timestamps land in training. Fit a tabular
# classifier and summarize checks.

from skore import TrainTestSplit, evaluate
from skrub import tabular_pipeline

splitter_shuffled = TrainTestSplit(random_state=42, shuffle=True)

report = evaluate(
    tabular_pipeline("classifier"),
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter_shuffled,
)
report

# %%
# SKD013 should list ``timestamp`` as overlapping between train and test.

report.checks.summarize()

# %%
report.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Optimistic scores under shuffled splits are a leakage artifact; SKD013 forces
# you to respect time ordering before trusting metrics.

# %%
# Chronological hold-out
# ======================
#
# ``shuffle=False`` keeps the test block as the latest rows in the table. No
# training row should carry a timestamp on or after the earliest test hire.

splitter_chrono = TrainTestSplit(random_state=42, shuffle=False)

report_chrono = evaluate(
    tabular_pipeline("classifier"),
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter_chrono,
)
report_chrono

# %%
# SKD013 should be absent; test rows are strictly after train rows.

report_chrono.checks.summarize(fast_mode=True)

# %%
report_chrono.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# A simple chronological hold-out is often enough for deployment monitoring
# when you score on the most recent period.

# %%
# TimeSeriesSplit for cross-validated evaluation
# ==============================================
#
# :class:`~sklearn.model_selection.TimeSeriesSplit` trains on past rows and
# tests on the next chunk in each fold. Many employees share the same hire
# date, so a default split can still place the same calendar day in train and
# test at the fold boundary (SKD013 uses ``>=``). A small ``gap`` skips rows
# between folds and clears that tie. Early folds are small and unrelated checks
# such as SKD008 may warn on encoded features; we ignore SKD008 here to focus
# on temporal validity.

from sklearn.model_selection import TimeSeriesSplit

splitter_tscv = TimeSeriesSplit(n_splits=5, gap=50)

report_tscv = evaluate(
    tabular_pipeline("classifier"),
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter_tscv,
)
report_tscv

# %%
# SKD013 should be absent; no fold trains on timestamps on or after its test
# block.

report_tscv.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
report_tscv.metrics.summarize(data_source="both").frame(aggregate="mean")

# %%
# Time-series cross-validation estimates stability across multiple forward
# windows, the right tool when a single hold-out is too noisy.

# %%
# Conclusion
# ==========
#
# SKD013 protects against training on the future. In this walkthrough,
# disabling shuffle and adopting ``TimeSeriesSplit`` aligned evaluation with
# how salary models are deployed over newly hired employees. Always pair
# temporal splits with features available at prediction time.
