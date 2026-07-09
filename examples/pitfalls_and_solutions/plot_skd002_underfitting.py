"""
.. _example_skd002_underfitting:

SKD002 — Potential underfitting
===============================

Reproduce :ref:`SKD002 <skd002-underfitting>` on the white wine quality dataset.

Mitigations from the :ref:`automated_checks` user guide:

- increase model capacity,
- improve data representation and features,
- tune hyperparameters,
- collect richer data if possible.
"""

# %%
# Load the white wine quality dataset
# ===================================

from sklearn.datasets import fetch_openml
from sklearn.linear_model import Ridge
from skore import TrainTestSplit, evaluate
from skrub import TableReport

RANDOM_STATE = 42

wine = fetch_openml(data_id=44971, as_frame=True, parser="auto")
df = wine.frame
X = df.drop(columns=["quality"])
y = df["quality"]

# %%
TableReport(X)

# %%
TableReport(y)

# %%
splitter = TrainTestSplit(random_state=RANDOM_STATE, shuffle=True)

# %%
# Trigger SKD002 — over-regularized Ridge
# =======================================
#
# When ``alpha`` is far too large: coefficients shrink toward zero and predictions
# stay near the mean. Train and test scores sit close to the dummy baseline.

over_regularized = Ridge(alpha=1_000_000)

report = evaluate(
    over_regularized,
    X=X,
    y=y,
    splitter=splitter,
)

report.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report.checks.summarize()
# Issue: SKD002 (potential underfitting) — scores on par with the dummy baseline.

# %%
# Tune hyperparameters
# ====================

import numpy as np
from sklearn.model_selection import GridSearchCV

tuned = GridSearchCV(Ridge(), param_grid={"alpha": np.logspace(-4, 4, 30)}, cv=3)

report_tuned = evaluate(tuned, X=X, y=y, splitter=splitter)

report_tuned.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_tuned.estimator_.best_params_

# %%
report_tuned.checks.summarize()
# SKD002 clears once ``alpha`` is in a reasonable range.

# %%
# Increase model capacity
# =======================
#
# A nonlinear tree ensemble can capture interactions a penalized linear model
# misses.

from skrub import tabular_pipeline

report_capacity = evaluate(
    tabular_pipeline("regressor"),
    X=X,
    y=y,
    splitter=splitter,
)

report_capacity.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_capacity.checks.summarize()
# SKD002 clears once the model has enough capacity to capture the signals.

# %%
# Improve data representation
# ===========================
#
# Scaling numeric inputs before Ridge often helps on tabular regression.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

report_scaled = evaluate(
    Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
    X=X,
    y=y,
    splitter=splitter,
)

report_scaled.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_scaled.checks.summarize()
# Scaling often clears SKD002 on this table.

# %%
# Collect richer data
# ===================
#
# Not demonstrated here: the OpenML table is fixed. In production you would
# add measurements that better explain wine quality.

# %%
# Compare mitigations
# ===================

from skore import compare

comparison = compare(
    {
        "over_regularized": report,
        "tuned_ridge": report_tuned,
        "tree_regressor": report_capacity,
        "scaled_ridge": report_scaled,
    }
)
comparison.metrics.summarize(data_source="both").frame(favorability=True)
