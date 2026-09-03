"""
.. _example_skd009_worse_than_baseline:

SKD009 - Model worse than baseline
==================================

This example walks through mitigations when check
:ref:`SKD009 <skd009-worse-than-baseline>` fires. The check trains a strong
:func:`~skrub.tabular_pipeline` baseline (gradient boosting on mixed tabular
data) and flags estimators that are not significantly better on default
metrics.

Mitigations from the :ref:`automated_checks` user guide, in the order we try
them here:

- revisit feature engineering and preprocessing,
- check whether the model family is appropriate,
- switch to a stronger default such as HistGradientBoostingRegressor,
- tune the model (here: moderated HGB capacity plus a log target).

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
# Trigger SKD009 with a linear pipeline
# =====================================
#
# Start with :func:`~skrub.tabular_pipeline` around
# :class:`~sklearn.linear_model.Ridge` so that encoding and imputation are
# already in place before fitting the linear model.

from sklearn.linear_model import Ridge
from skore import evaluate
from skrub import tabular_pipeline

report_ridge = evaluate(
    tabular_pipeline(Ridge()),
    X=X,
    y=y,
    splitter=splitter,
)
report_ridge

# %%
# Find ``SKD009`` in the Tips tab below: the Ridge pipeline should report
# worse-than-baseline performance on a majority of metrics.

report_ridge.checks.summarize()

# %%
report_ridge.metrics.summarize(data_source="both").frame()

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
# Keep everything inside a pipeline so the same transforms run at predict time
# on new data.

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


def engineer_features(X):
    drg = X["DRG_Definition"].str.upper()
    return X.drop(columns=["Total_Discharges"]).assign(
        log_Total_Discharges=np.log1p(X["Total_Discharges"]),
        DRG_Code=(
            X["DRG_Definition"].str.extract(r"^(\d+)", expand=False).astype(float)
        ),
        has_MCC=drg.str.contains("W MCC", regex=False).astype(int),
        has_CC=(
            drg.str.contains("W CC", regex=False)
            & ~drg.str.contains("W MCC", regex=False)
        ).astype(int),
    )


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
report_ridge_fe

# %%
# Feature engineering may improve metrics while SKD009 still flags a linear
# model that cannot match the HGB baseline on every score.

report_ridge_fe.checks.summarize(fast_mode=True)

# %%
report_ridge_fe.metrics.summarize(data_source="both").frame()

# %%
# Check model family with a random forest
# =======================================
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
report_rf

# %%
comparison_families = compare(
    {
        "ridge_with_fe": report_ridge_fe,
        "random_forest": report_rf,
    }
)
comparison_families.metrics.summarize(data_source="both").frame()

# %%
report_rf.checks.summarize(fast_mode=True)

# %%
# Switch to HistGradientBoostingRegressor
# =======================================
#
# skore's SKD009 performance baseline is itself an HGB pipeline. Matching that
# family is the natural next step once trees look promising, but defaults are
# not guaranteed to clear the check, because SKD009 asks whether you are
# *significantly* better than a strong HGB baseline.

from sklearn.ensemble import HistGradientBoostingRegressor

report_hgb = evaluate(
    tabular_pipeline(HistGradientBoostingRegressor(random_state=42)),
    X=X,
    y=y,
    splitter=splitter,
)
report_hgb

# %%
report_hgb.checks.summarize(fast_mode=True)

# %%
report_hgb.metrics.summarize(data_source="both").frame()

# %%
# Combine levers: features, HGB, and a log target
# ===============================================
#
# In practice you would usually tune these knobs with RandomizedSearchCV or
# GridSearchCV from scikit-learn. To keep the example short and reproducible, we
# pin one search outcome that clears SKD009 on this split by stacking the
# earlier levers:
#
# - the engineered features,
# - an HGB with moderated capacity (learning rate, leaf size, ``l2``),
# - :class:`~sklearn.compose.TransformedTargetRegressor` with ``log1p`` /
#   ``expm1``, because payment totals are heavy-tailed.
#
# SKD009 needs a *significant* win over default HGB, not merely matching it, so
# representation, family, and this tuned setup matter together.

from sklearn.compose import TransformedTargetRegressor

tuned = TransformedTargetRegressor(
    regressor=make_pipeline(
        FunctionTransformer(engineer_features),
        tabular_pipeline(
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=500,
                max_depth=5,
                max_leaf_nodes=63,
                min_samples_leaf=10,
                l2_regularization=0.1,
                random_state=42,
            )
        ),
    ),
    func=np.log1p,
    inverse_func=np.expm1,
)

report_tuned = evaluate(tuned, X=X, y=y, splitter=splitter)
report_tuned

# %%
# SKD009 should clear.

report_tuned.checks.summarize()

# %%
report_tuned.metrics.summarize(data_source="both").frame()

# %%
# Conclusion
# ==========
#
# SKD009 guards against estimators that underperform a strong tabular baseline.
# Clearing it was not a single knob: feature engineering, a tree family,
# matching HGB, *and* a tuned setup (including a log target for skewed payments)
# together pushed past the check. Prefer starting from skrub's
# :func:`~skrub.tabular_pipeline`, then combine features, family, and
# hyperparameters until checks and business metrics align.
