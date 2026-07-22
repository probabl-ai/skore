"""
.. _example_skd009_worse_than_baseline:

SKD009 — Model worse than baseline
==================================

This example walks through mitigations when check
:ref:`SKD009 <skd009-worse-than-baseline>` fires. The check trains a strong
:func:`~skrub.tabular_pipeline` baseline — gradient boosting on mixed tabular
data — and flags estimators that are not significantly better on default
metrics.

Mitigations from the :ref:`automated_checks` user guide, in the order we try
them here:

- revisit feature engineering and preprocessing,
- check whether the model family is appropriate,
- switch to a stronger default such as HistGradientBoostingRegressor,
- tune the model's hyperparameters.

We use the medical charge dataset with provider IDs and leakage columns removed.
The goal is to beat skore's HGB baseline on held-out payment totals.
"""

# %%
# Load the medical charge dataset
# ===============================
#
# :func:`skrub.datasets.fetch_medical_charge` returns hospital billing records.
# We drop provider identifiers and columns that leak the target, then subsample
# 2,000 rows for a challenging regression task.
from skrub.datasets import fetch_medical_charge

dataset = fetch_medical_charge()
X_full, y_full = dataset.X, dataset.y

id_cols = [
    "Provider_Zip_Code",
    "Provider_Id",
    "Provider_Name",
    "Provider_Street_Address",
]
leakage_cols = ["Average_Covered_Charges", "Average_Medicare_Payments"]

X = X_full.drop(columns=id_cols + leakage_cols).sample(2_000, random_state=42)
y = y_full.loc[X.index]

# %%
# Inspect the features matrix with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
# The target is continuous total payment.

TableReport(y)

# %%
from skore import TrainTestSplit

splitter = TrainTestSplit(random_state=42, test_size=0.2)

# %%
# Trigger SKD009 — linear pipeline
# ================================
#
# Start with a linear baseline:
# :func:`~skrub.tabular_pipeline` around :class:`~sklearn.linear_model.Ridge`.

from sklearn.linear_model import Ridge
from skore import evaluate
from skrub import tabular_pipeline

report_ridge = evaluate(
    tabular_pipeline(Ridge()),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# SKD009 should report worse-than-baseline performance on a majority of metrics.

report_ridge.checks.summarize()

# %%
report_ridge.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Revisit feature engineering
# ===========================
#
# A linear model benefits from spelling out structure that trees would discover
# from the raw columns:
#
# - ``log1p(Total_Discharges)`` compresses a skewed volume signal (then drop the
#   raw count to avoid collinearity),
# - extract the numeric DRG code from labels like ``"178 - ... W CC"``,
# - flag severity markers in the text (``W MCC`` / ``W CC``), which affect
#   payment levels.
#
# Keep everything inside a pipeline so the same transforms run at predict time.

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


def engineer_features(X):
    X = X.copy()
    X["log_Total_Discharges"] = np.log1p(X["Total_Discharges"])
    X = X.drop(columns=["Total_Discharges"])
    X["DRG_Code"] = (
        X["DRG_Definition"].str.extract(r"^(\d+)", expand=False).astype(float)
    )
    drg = X["DRG_Definition"].str.upper()
    X["has_MCC"] = drg.str.contains("W MCC", regex=False).astype(int)
    X["has_CC"] = (
        drg.str.contains("W CC", regex=False) & ~drg.str.contains("W MCC", regex=False)
    ).astype(int)
    return X


ridge_with_fe = make_pipeline(
    FunctionTransformer(engineer_features),
    tabular_pipeline(Ridge()),
)

report_ridge_fe = evaluate(
    ridge_with_fe,
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# Feature engineering may improve metrics while SKD009 still flags a linear
# model that cannot match the HGB baseline on every score.

report_ridge_fe.checks.summarize()

# %%
report_ridge_fe.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Check model family — random forest
# ==================================
#
# If nonlinearity and interactions matter, trees should close much of the gap.
# Compare a :class:`~sklearn.ensemble.RandomForestRegressor` pipeline to the
# engineered Ridge on the same split.

from sklearn.ensemble import RandomForestRegressor
from skore import compare

report_rf = evaluate(
    tabular_pipeline(
        RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=4,
        )
    ),
    X=X,
    y=y,
    splitter=splitter,
)

comparison_families = compare(
    {
        "ridge_with_fe": report_ridge_fe,
        "random_forest": report_rf,
    }
)
comparison_families.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_rf.checks.summarize()

# %%
# Switch to HistGradientBoostingRegressor
# =======================================
#
# skore's SKD009 performance baseline is itself an HGB pipeline. Matching that
# family is the natural next step once trees look promising — but defaults are
# not guaranteed to clear the check, because SKD009 asks whether you are
# *significantly* better than a strong HGB baseline.

from sklearn.ensemble import HistGradientBoostingRegressor

report_hgb = evaluate(
    tabular_pipeline(HistGradientBoostingRegressor(random_state=42)),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
report_hgb.checks.summarize()

# %%
report_hgb.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Tune hyperparameters
# ====================
#
# Tuning alone is rarely enough: the effort that clears SKD009 comes from
# *combining* the earlier levers — engineered features, a strong tree family,
# and then hyperparameter search. Fold the feature engineering into an HGB
# :class:`~sklearn.model_selection.RandomizedSearchCV` over leaf size, depth,
# learning rate, and iteration budget.

from sklearn.model_selection import RandomizedSearchCV

hgb_with_fe = make_pipeline(
    FunctionTransformer(engineer_features),
    tabular_pipeline(HistGradientBoostingRegressor(random_state=42)),
)

param_distributions = {
    "pipeline__histgradientboostingregressor__max_depth": [None, 5, 8, 15],
    "pipeline__histgradientboostingregressor__learning_rate": [0.05, 0.1, 0.15],
    "pipeline__histgradientboostingregressor__max_iter": [100, 200, 300, 500],
    "pipeline__histgradientboostingregressor__max_leaf_nodes": [15, 31, 63, 127],
    "pipeline__histgradientboostingregressor__min_samples_leaf": [5, 10, 20],
}

tuned = RandomizedSearchCV(
    hgb_with_fe,
    param_distributions=param_distributions,
    n_iter=12,
    cv=3,
    random_state=42,
    n_jobs=4,
)

report_tuned = evaluate(tuned, X=X, y=y, splitter=splitter)

# %%
# Stacking representation, family, and tuning should clear SKD009.

report_tuned.checks.summarize()

# %%
report_tuned.estimator_.best_params_

# %%
report_tuned.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Conclusion
# ==========
#
# SKD009 guards against estimators that underperform a strong tabular baseline.
# Clearing it here was not a single knob: feature engineering, choosing an
# appropriate tree family, matching the HGB baseline, and tuning that pipeline
# together pushed past the check. Start with :func:`~skrub.tabular_pipeline`,
# then combine features, family, and hyperparameters until checks and business
# metrics align.
