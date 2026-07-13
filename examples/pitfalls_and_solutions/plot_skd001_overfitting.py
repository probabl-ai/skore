"""
.. _example_skd001_overfitting:

SKD001 — Potential overfitting
==============================

When a model fits the training set much better than the hold-out test set, it may
have memorized patterns of the training data instead of generalizing.
:ref:`SKD001 <skd001-overfitting>` flags this situation by comparing train and test
scores across the report's default metrics.

This example walks through practical mitigations suggested in
:ref:`automated_checks`:

- regularize more strongly,
- improve feature engineering,
- use better validation protocols or more data.

We use the California housing dataset to predict median house values (in k$).
A default :class:`~sklearn.ensemble.RandomForestRegressor` overfits on half of the
rows; we keep the other half to show how more data, combined with regularization
and richer features, narrows the train/test gap.
"""

# %%
# Load the California housing dataset
# ===================================
#
# Each row describes a block in California. The target is ``MedHouseVal``
# \$100k; we multiply by 100 so errors read naturally in k$. We use half
# of the rows (~10k) for the main example.

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing(as_frame=True)
y_full = housing.target * 100

X, X_heldout, y, y_heldout = train_test_split(
    housing.data,
    y_full,
    train_size=0.5,
    random_state=42,
)

X = X.reset_index(drop=True)
y = pd.Series(y, name="MedHouseVal_k$").reset_index(drop=True)
X_heldout = X_heldout.reset_index(drop=True)
y_heldout = pd.Series(y_heldout, name="MedHouseVal_k$").reset_index(drop=True)

# %%
# :class:`~skrub.TableReport` helps explore the feature matrix and target more easily.
from skrub import TableReport

TableReport(X)

# %%

TableReport(y)

# %%
# A single random train-test split is used in the :func:`~skore.evaluate` cells below.

from skore import TrainTestSplit

splitter = TrainTestSplit(random_state=42)

# %%
# Trigger SKD001 — default random forest
# ======================================
#
# :class:`~sklearn.ensemble.RandomForestRegressor` with default settings tends to
# overfit tabular data: train scores look strong while test scores lag behind.

from sklearn.ensemble import RandomForestRegressor
from skore import evaluate

report = evaluate(
    RandomForestRegressor(random_state=42),
    X=X,
    y=y,
    splitter=splitter,
)

report.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# :meth:`~skore.EstimatorReport.checks.summarize` should report SKD001.

report.checks.summarize()

# %%
# Regularize more strongly
# ========================
#
# The model is regularized by limiting leaf size and the number of features considered at each split.
# This helps curb memorization while keeping test performance high.

report_regularized = evaluate(
    RandomForestRegressor(
        min_samples_leaf=10,
        max_features=0.5,
        random_state=42,
    ),
    X=X,
    y=y,
    splitter=splitter,
)

report_regularized.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Train scores move closer to test scores. The gap narrows even if SKD001 may
# still fire until the other mitigations below are applied.

report_regularized.checks.summarize()

# %%
# Improve feature engineering
# ===========================
#
# House prices here depend on how income, occupancy, and location combine: the same
# median income can imply very different values depending on where a block sits, so
# linear effects of individual columns are not enough. Following the ideas in
# :ref:`example_feature_importance`, we first replace ``Latitude`` and ``Longitude``
# with cluster labels from :class:`~sklearn.cluster.KMeans`, grouping nearby blocks
# into a handful of regions. We then add degree-2 interaction terms with
# :class:`~sklearn.preprocessing.PolynomialFeatures` so the forest can learn
# region-specific income and occupancy effects instead of treating every feature on
# its own.

from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

geo_columns = ["Latitude", "Longitude"]
preprocessor = make_column_transformer(
    (KMeans(n_clusters=10, random_state=0), geo_columns),
    remainder="passthrough",
)
engineered_rf = make_pipeline(
    preprocessor,
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    RandomForestRegressor(random_state=42),
)

report_engineered = evaluate(
    engineered_rf,
    X=X,
    y=y,
    splitter=splitter,
)

report_engineered.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Test scores typically improve once the model can use interactions between
# income, occupancy, and location. The train/test gap also narrows compared with
# the default forest on raw columns.

report_engineered.checks.summarize()

# %%
# Use more training data
# ======================
#
# We hold out one fixed test set, then fit the engineered pipeline on the
# original training fold and on the same fold augmented with ``X_heldout``.
# We also switch to :class:`~sklearn.ensemble.HistGradientBoostingRegressor`,
# a more powerful model to show that adding the held-out data reduces overfit
# significantly.
from sklearn.ensemble import HistGradientBoostingRegressor
from skore import EstimatorReport, compare

engineered_hgbr = make_pipeline(
    preprocessor,
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    HistGradientBoostingRegressor(random_state=42),
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)  # same 20% hold-out as ``splitter``

X_train_augmented = pd.concat([X_train, X_heldout])
y_train_augmented = pd.concat([y_train, y_heldout])

report_less_data = EstimatorReport(
    engineered_hgbr,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

report_more_data = EstimatorReport(
    engineered_hgbr,
    X_train=X_train_augmented,
    y_train=y_train_augmented,
    X_test=X_test,
    y_test=y_test,
)

# %%
# Test error typically improves when the model sees the held-out rows during
# training, even though train scores remain optimistic.

report_more_data.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_more_data.checks.summarize()
# %%
# Compare mitigations
# ===================
#
# :func:`~skore.compare` lines up every report on a shared test set so we can
# read off how each lever moved the metrics.

comparison = compare(
    {
        "default_rf": report,
        "regularized_rf": report_regularized,
        "engineered_rf": report_engineered,
        "less_training_data": report_less_data,
        "more_training_data": report_more_data,
    }
)
comparison.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Conclusion
# ==========
#
# SKD001 warns against models that excel on training data but underperform on
# hold-out rows. Here, a default random forest on California housing showed a
# wide train/test gap; regularization, richer features, and more training data
# each narrowed it. Combine these levers in practice until checks clear.
