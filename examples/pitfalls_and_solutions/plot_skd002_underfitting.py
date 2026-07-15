"""
.. _example_skd002_underfitting:

SKD002 — Potential underfitting
===============================

When train and test scores are close to a dummy baseline, the model is not learning
enough structure from the inputs. :ref:`SKD002 <skd002-underfitting>` flags
this pattern across the report's default predictive metrics.

This example walks through mitigations from :ref:`automated_checks`:

- increase model capacity,
- improve data representation and features,
- tune hyperparameters,
- collect richer data if possible.

We use the California housing regression task: predict median house values from
census block attributes on their natural scales (income, room counts,
population, latitude, and longitude). Starting from a mean-predicting dummy, we
add model capacity, engineer a more informative feature for the linear model,
and tune a nonlinear regressor while keeping the same validation split.
"""

# %%
# Load the California housing dataset
# ===================================
#
# Each row describes a block group in California. The target is ``MedHouseVal``
# in \$100k units. Features such as income, room counts, population, and
# coordinates are on their natural scales rather than pre-centered.

from skrub.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y = housing.X, housing.y

# %%
# skrub's :class:`~skrub.TableReport` is used to visualize the data.
from skrub import TableReport

TableReport(X)

# %%

TableReport(y)

# %%
# A single shuffled train-test split is used in the :func:`~skore.evaluate` cells
# below.

from skore import TrainTestSplit

splitter = TrainTestSplit(random_state=42, shuffle=True)

# %%
# Trigger SKD002 — dummy baseline
# ===============================
#
# :class:`~sklearn.dummy.DummyRegressor` always predicts the training mean. It
# is the same naive baseline that SKD002 compares against, so train and test
# scores stay on par with it.

from sklearn.dummy import DummyRegressor
from skore import evaluate

report_dummy = evaluate(
    DummyRegressor(),
    X=X,
    y=y,
    splitter=splitter,
)

report_dummy.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# :meth:`~skore.EstimatorReport.checks.summarize` should report SKD002.

report_dummy.checks.summarize()

# %%
# Increase model capacity — Ridge
# ===============================
#
# A standard :class:`~sklearn.linear_model.Ridge` regressor is enough to clear
# SKD002: test :math:`R^2` moves clearly above the mean-predicting dummy.

from sklearn.linear_model import Ridge

report_ridge = evaluate(
    Ridge(),
    X=X,
    y=y,
    splitter=splitter,
)

report_ridge.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_ridge.checks.summarize()

# %%
# Increase model capacity — gradient boosting
# ============================================
#
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor` adds nonlinear
# capacity. It can capture interactions between income, room counts, and
# location that a linear model misses, often lifting test scores further.

from sklearn.ensemble import HistGradientBoostingRegressor

report_hgbr = evaluate(
    HistGradientBoostingRegressor(random_state=42),
    X=X,
    y=y,
    splitter=splitter,
)

report_hgbr.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Compare train and test rows: a wide gap here can hint at overfitting even
# after SKD002 has cleared.

report_hgbr.checks.summarize()

# %%
# Improve data representation
# ===========================
#
# Raw room and occupancy counts do not always line up with how spacious a block
# feels. Dividing ``AveRooms`` by ``AveOccup`` yields an average number of
# rooms per person — a better signal for a linear model. We keep the original
# columns and fit the same :class:`~sklearn.linear_model.Ridge` on the
# enriched table.

X_engineered = X.assign(RoomsPerPerson=X["AveRooms"] / X["AveOccup"].clip(lower=0.1))

report_engineered_ridge = evaluate(
    Ridge(),
    X=X_engineered,
    y=y,
    splitter=splitter,
)

report_engineered_ridge.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_engineered_ridge.checks.summarize()

# %%
# Tune hyperparameters
# ====================
#
# Grid search over tree depth, leaf size, and learning rate regularizes the
# ensemble while keeping its nonlinear capacity. On this table it refines the
# default booster rather than replacing it.

from sklearn.model_selection import GridSearchCV

report_hgbr_tuned = evaluate(
    GridSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        param_grid={
            "max_depth": [3, 4, 6],
            "min_samples_leaf": [10, 15, 20, 30],
            "learning_rate": [0.05, 0.1, 0.5],
        },
        cv=3,
    ),
    X=X,
    y=y,
    splitter=splitter,
)

report_hgbr_tuned.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_hgbr_tuned.estimator_.best_params_

# %%
report_hgbr_tuned.checks.summarize()

# %%
# Collect richer data
# ===================
#
# The California housing table is fixed in this example. In production you
# would add measurements that better explain prices — school quality, crime
# rates, renovation history, or commute times — so the model has more signal
# to learn from.

# %%
# Compare mitigations
# ===================
#
# :func:`~skore.compare` lines up every report on the shared test fold so you
# can read off how each lever moved the metrics.

from skore import compare

comparison = compare(
    {
        "dummy": report_dummy,
        "ridge": report_ridge,
        "hgbr": report_hgbr,
        "ridge_engineered": report_engineered_ridge,
        "hgbr_tuned": report_hgbr_tuned,
    }
)
comparison.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Conclusion
# ==========
#
# SKD002 warns when a model barely beats a mean predictor. Here, a dummy
# regressor triggered the check and a plain Ridge cleared it. A gradient
# booster added nonlinear capacity above the linear baseline, a rooms-per-person
# feature lifted the linear model further, and hyperparameter search refined the
# ensemble. Combine these levers in practice, and revisit data collection when
# the feature table itself is the bottleneck.
