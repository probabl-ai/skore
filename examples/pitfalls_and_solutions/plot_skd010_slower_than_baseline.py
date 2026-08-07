"""
.. _example_skd010_slower_than_baseline:

SKD010 — Model slower than baseline
===================================

This example walks through mitigations when the check
:ref:`SKD010 <skd010-slower-than-baseline>` fires. The check compares the user's model
to a fast linear baseline — :class:`~sklearn.linear_model.RidgeCV` for
regression, wrapped in :func:`~skrub.tabular_pipeline` — and flags a problem
only when both of the following hold:

- fit time is at least 2 times the baseline (and at least 0.05s slower), and
- test scores are not significantly better than the baseline on a majority
  of default predictive metrics.

A slow model that clearly beats RidgeCV on quality does not trigger SKD010; a
fast model that ties RidgeCV does not either. The issue is paying for latency
without a significant quality premium.

Mitigations from the :ref:`automated_checks` user guide, in the order we try
them here:

- check that preprocessing is not the actual bottleneck,
- reduce the model's complexity,
- switch to the fast linear baseline when quality is sufficient,
- profile fit time to understand the dominant cost.

We use the employee salaries dataset with a heavy random forest inside
:func:`~skrub.tabular_pipeline`. The goal is either to match RidgeCV quality at
lower cost, or to shrink fit time until the speed gap is no longer unjustified.
"""

# %%
# Load the employee salaries dataset
# ==================================
#
# :func:`skrub.datasets.fetch_employee_salaries` returns human-resources records
# with mixed categorical and numeric fields. A 200-tree random forest inside
# ``tabular_pipeline`` trains slowly yet often fails to beat the fast RidgeCV
# baseline on test scores. SKD010 is a slow check.

from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
X = dataset.X
y = dataset.y.squeeze()

# %%
# Inspect column types with :class:`~skrub.TableReport` — encoding cost is part
# of total fit time.

from skrub import TableReport

TableReport(X)

# %%
# Salaries are continuous and right-skewed, a typical regression target.

TableReport(y.to_frame())

# %%
from skore import TrainTestSplit

splitter = TrainTestSplit(test_size=0.2, random_state=42)

# %%
# Trigger SKD010 — heavy random forest pipeline
# =============================================
#
# Two hundred shallow trees with full tabular preprocessing are deliberately
# expensive. Expect SKD010 when this pipeline is much slower than RidgeCV and
# its test metrics are not significantly better.

from sklearn.ensemble import RandomForestRegressor
from skore import evaluate
from skrub import tabular_pipeline

report = evaluate(
    tabular_pipeline(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            random_state=42,
            n_jobs=4,
        )
    ),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# SKD010 should report a large fit-time ratio without a significant quality win.

report.checks.summarize()

# %%
report.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Check whether preprocessing is the bottleneck
# =============================================
#
# First ask whether categorical encoding is what makes the pipeline slow. Fit
# the same 200-tree forest on numeric columns only, skipping that encoding.
#
# SKD010 can still fire here: 200 trees may remain more than 2 times slower than RidgeCV
# on the numeric table, and without categoricals the model is often not
# significantly better. If the numeric-only forest stays slow, encoding was not
# the dominant cost — the forest itself was.

numeric_cols = X.select_dtypes(include="number").columns.tolist()
X_numeric = X[numeric_cols]

report_numeric = evaluate(
    RandomForestRegressor(
        n_estimators=200,
        max_depth=3,
        random_state=42,
        n_jobs=4,
    ),
    X=X_numeric,
    y=y,
    splitter=splitter,
)

# %%
report_numeric.checks.summarize()

# %%
report_numeric.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# SKD010 did not drop after removing categoricals. Absolute fit time may look
# smaller, but the forest is still much slower than the RidgeCV baseline on the
# numeric table without a significant quality win. Encoding was therefore not
# the problem — the model was.

# %%
# Reduce model complexity
# =======================
#
# Fewer estimators cut fit time. SKD010 often clears here because the slowness
# gate fails (no longer more than 2 times RidgeCV), even if test scores stay
# comparable; the check needs both slowness and no significant quality gain.

report_lighter = evaluate(
    tabular_pipeline(
        RandomForestRegressor(
            n_estimators=25,
            max_depth=3,
            random_state=42,
            n_jobs=4,
        )
    ),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# With a lighter forest, SKD010 should typically be gone.

report_lighter.checks.summarize()

# %%
report_lighter.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Switch to the fast linear baseline
# ==================================
#
# :class:`~sklearn.linear_model.RidgeCV` inside ``tabular_pipeline`` is the
# reference skore uses for regression speed. Adopting it clears SKD010: you are
# no longer slower than the baseline you are compared against.

from sklearn.linear_model import RidgeCV

report_linear = evaluate(
    tabular_pipeline(RidgeCV()),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# SKD010 should be absent — this pipeline is the fast baseline.

report_linear.checks.summarize()

# %%
report_linear.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Profile fit times
# =================
#
# Compare train fit time (``_fit_time``) and test predict time for the
# estimators we keep in the mitigation path. Cutting ``n_estimators`` is what
# often exits the 2× gate; RidgeCV is the speed reference.

import pandas as pd
from skore import compare

pd.DataFrame(
    {
        "heavy_rf_pipeline": [
            report._fit_time,
            report.metrics.predict_time(data_source="test"),
        ],
        "lighter_rf_pipeline": [
            report_lighter._fit_time,
            report_lighter.metrics.predict_time(data_source="test"),
        ],
        "ridge_pipeline": [
            report_linear._fit_time,
            report_linear.metrics.predict_time(data_source="test"),
        ],
    },
    index=["fit_time_train_s", "predict_time_test_s"],
).T

# %%
# Compare predictive metrics
# ==========================
#
# Look at test scores together with the timings above. SKD010 cares about whether
# extra seconds buy a *significant* score gain over RidgeCV.

comparison = compare(
    {
        "heavy_rf_pipeline": report,
        "lighter_rf_pipeline": report_lighter,
        "ridge_pipeline": report_linear,
    }
)
comparison.metrics.summarize(data_source="test").frame(favorability=True)

# %%
# Conclusion
# ==========
#
# SKD010 fires only for models that are much slower than a fast linear baseline
# and not significantly better on test metrics. The numeric-only ablation
# showed preprocessing need not be the bottleneck; fewer trees often clear the
# check by exiting the 2× gate; RidgeCV removes it by matching the speed
# baseline. Prefer a lighter model or the fast baseline unless held-out scores
# clearly pay for the extra fit time.
