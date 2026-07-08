"""
.. _example_skd001_overfitting:

SKD001 — Potential overfitting
==============================

Reproduce :ref:`SKD001 <skd001-overfitting>` on the OpenML electricity dataset.

Mitigations from the :ref:`automated_checks` user guide:

- regularize more strongly,
- simplify the model,
- improve feature engineering,
- use better validation protocols or more data.
"""

# %%
# Load the electricity dataset
# ============================
#
# Rows are time-ordered. ``shuffle=False`` keeps the hold-out as future data,
# a realistic split for this table.

import openml
import pandas as pd
from skore import TrainTestSplit, evaluate
from skrub import TableReport, tabular_pipeline

RANDOM_STATE = 42
POS_LABEL = "UP"

dataset = openml.datasets.get_dataset("electricity")
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# %%
TableReport(X)

# %%
TableReport(y)

# %%
splitter = TrainTestSplit(random_state=RANDOM_STATE, shuffle=False)
splitter_more_data = TrainTestSplit(
    test_size=0.1,
    random_state=RANDOM_STATE,
    shuffle=False,
)

# %%
# Trigger SKD001 — untuned gradient boosting
# ==========================================
#
# ``tabular_pipeline("classifier")`` has enough capacity to overfit on a single
# hold-out. Compare train vs test in the metrics table, then run checks.

from sklearn.ensemble import HistGradientBoostingClassifier

estimator = tabular_pipeline(HistGradientBoostingClassifier(random_state=RANDOM_STATE))

report = evaluate(
    estimator,
    X=X,
    y=y,
    pos_label=POS_LABEL,
    splitter=splitter,
)

report.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report.checks.summarize()
# Issue: SKD001 (potential overfitting) — train scores clearly beat test scores.

# %%
# Regularize more strongly
# ========================
#
# Lower depth, bigger leaves, and a slower learning rate curb memorization.

report_regularized = evaluate(
    tabular_pipeline(
        HistGradientBoostingClassifier(
            max_depth=3,
            min_samples_leaf=50,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        )
    ),
    X=X,
    y=y,
    pos_label=POS_LABEL,
    splitter=splitter,
)

report_regularized.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_regularized.checks.summarize()
# SKD001 often clears once the train/test gap narrows.

# %%
# Simplify the model
# ==================
#
# A linear model has less capacity to memorize training quirks.

from sklearn.linear_model import LogisticRegression

report_simple = evaluate(
    tabular_pipeline(LogisticRegression(max_iter=10_000)),
    X=X,
    y=y,
    pos_label=POS_LABEL,
    splitter=splitter,
)

report_simple.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_simple.checks.summarize()
# The gap often shrinks; SKD001 may still fire depending on the split.

# %%
# Use more training data
# ======================
#
# A smaller hold-out leaves more rows for fitting — sometimes enough to reduce
# overfitting without changing the estimator.

report_more_data = evaluate(
    estimator,
    X=X,
    y=y,
    pos_label=POS_LABEL,
    splitter=splitter_more_data,
)

report_more_data.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_more_data.checks.summarize()
# More training rows help reduce the gap but do not always clear SKD001 on their
# own.

# %%
# Improve feature engineering
# ===========================
#
# A per-row ``row_id`` is pure noise but lets the tree memorize rows (train
# accuracy inflates). We add it first to worsen overfitting, then drop it along
# with ``vicprice`` that has little to no signal. (Verify with the inspection
# module # ``report.inspection.impurity_decrease().plot()``)

n_samples = len(X)
row_id = pd.Series(
    [f"row_{i}" for i in range(n_samples)],
    index=X.index,
    name="row_id",
).astype("category")
X_spurious = pd.concat([X, row_id], axis=1)

report_spurious = evaluate(
    estimator,
    X=X_spurious,
    y=y,
    pos_label=POS_LABEL,
    splitter=splitter,
)

report_spurious.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_spurious.checks.summarize()
# SKD001 is often stronger once spurious columns are in the table.

# %%
X_engineered = X_spurious.drop(columns=["row_id", "vicprice"])

report_fe = evaluate(
    estimator,
    X=X_engineered,
    y=y,
    pos_label=POS_LABEL,
    splitter=splitter,
)

report_fe.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_fe.checks.summarize()
# Dropping spurious columns narrows the gap vs ``report_spurious``.

# %%
# Compare mitigations
# ===================
#
# ``compare()`` needs the same test set — ``report_more_data`` used a different
# split, so we compare it separately above.

from skore import compare

comparison = compare(
    {
        "default": report,
        "regularized": report_regularized,
        "simpler": report_simple,
    }
)
comparison.metrics.summarize(data_source="both").frame(favorability=True)

# %%
comparison_fe = compare({"with_row_id": report_spurious, "engineered": report_fe})
comparison_fe.metrics.summarize(data_source="both").frame(favorability=True)
