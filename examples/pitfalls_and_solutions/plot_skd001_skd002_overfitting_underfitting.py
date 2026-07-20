"""
.. _example_skd001_skd002_overfitting_underfitting:

SKD001 & SKD002 — Overfitting and underfitting
==============================================

:ref:`SKD002 <skd002-underfitting>` and :ref:`SKD001 <skd001-overfitting>`
describe opposite ends of the same problem: model *expressiveness*.

- Too little expressiveness: train and test scores stay close to a weak
  learner / dummy baseline (SKD002).
- Too much expressiveness: train scores pull far ahead of test scores
  (SKD001).

The correct the model beats that baseline on both train and test, without
memorizing the training set. This notebook walks that path while showing
different mitigations

- capacity (model family / complexity)
- features
- regularization
- early stopping
- more data

We use California housing (median house value in k\$). Half of the rows feed
the main walkthrough; the other half is reserved to show the effect of adding
training data.
"""

# %%
# Load the California housing dataset
# ===================================
#
# Each row is a census block group. The target ``MedHouseVal`` is in \$100k
# units; we multiply by 100 so errors read in k\$.

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

y = pd.Series(y, name="MedHouseVal_k$")
y_heldout = pd.Series(y_heldout, name="MedHouseVal_k$")

# %%
from skrub import TableReport

TableReport(X)

# %%
TableReport(y)

# %%
# We use the same split for every :func:`~skore.evaluate` call.

from skore import TrainTestSplit, compare, evaluate

splitter = TrainTestSplit(test_size=0.2, random_state=42)

# %%
# Underfitting: SKD002 fires
# ==========================
#
# We start with a linear model that is overly regularized.
# :class:`~sklearn.linear_model.Ridge` with a very large ``alpha`` shrinks
# coefficients toward zero, so predictions stay close to a dummy baseline.

from sklearn.linear_model import Ridge

report_underfit = evaluate(
    Ridge(alpha=1_000_000),
    X=X,
    y=y,
    splitter=splitter,
)
report_underfit.metrics.summarize(data_source="both").frame()

# %%
# SKD002 should be present.

report_underfit.checks.summarize(fast_mode=True)

# %%
# Increase model capacity
# =======================
#
# Moving away from underfitting means giving the model enough expressiveness to
# use the inputs. Dropping to a default :class:`~sklearn.linear_model.Ridge`
# (mild regularization) already learns useful weights on each feature and is
# usually enough to clear SKD002 on this table.

report_ridge = evaluate(Ridge(), X=X, y=y, splitter=splitter)
report_ridge.metrics.summarize(data_source="both").frame()

# %%
report_ridge.checks.summarize(fast_mode=True)

# %%
# A :class:`~sklearn.ensemble.HistGradientBoostingRegressor` adds nonlinear
# capacity and can lift test scores further. Watch the train/test gap as you do
# this: the same lever that cures underfitting is the one that creates
# overfitting.

from sklearn.ensemble import HistGradientBoostingRegressor

report_hgbr = evaluate(
    HistGradientBoostingRegressor(random_state=42),
    X=X,
    y=y,
    splitter=splitter,
)
report_hgbr.metrics.summarize(data_source="both").frame()

# %%
report_hgbr.checks.summarize(fast_mode=True)

# %%
# Feature engineering
# ===================
#
# Better features help when the model family is fine but the *representation*
# is weak (still an underfitting problem). Here, ``AveRooms / AveOccup`` is a
# rooms-per-person signal that a linear model can use more easily than the raw
# counts.
#
# The same step has an overfitting side: every new feature also increases
# expressiveness. Mild engineering can close an underfit gap; aggressive
# expansions (very high-degree interactions, huge one-hot spaces) can reopen an
# overfit gap. Think of features as capacity you add to the *inputs*, not only
# to the estimator.

X_fe = X.assign(RoomsPerPerson=X["AveRooms"] / X["AveOccup"].clip(lower=0.1))

report_ridge_fe = evaluate(Ridge(), X=X_fe, y=y, splitter=splitter)
report_ridge_fe.metrics.summarize(data_source="both").frame()

# %%
report_ridge_fe.checks.summarize(fast_mode=True)

# %%
# Overfitting: SKD001 fires
# =========================
#
# To see the other end of the continuum on the **same** split, fit a default
# :class:`~sklearn.ensemble.RandomForestRegressor`. Unrestricted leaves can
# memorize training idiosyncrasies: train metrics look excellent, test metrics
# lag, and SKD001 flags the gap.

from sklearn.ensemble import RandomForestRegressor

report_rf = evaluate(
    RandomForestRegressor(random_state=42),
    X=X,
    y=y,
    splitter=splitter,
)
report_rf.metrics.summarize(data_source="both").frame()

# %%
report_rf.checks.summarize(fast_mode=True)

# %%
# Regularization
# ==============
#
# Once SKD001 appears, you pull expressiveness back with capacity limits on the
# estimator (leaf size, feature fraction, depth, learning rate, …).
#
# Here, we set the hyperparameters of the model by hand. In practice, it is
# rather hard to know in advance which combination of hyperparameters leads to
# generalization. One should lean towards tuning hyperparameters. We would
# advocate for randomized search, with successive halving when the amount of
# samples is large. See `tuning the hyper-parameters of an estimator
# <https://scikit-learn.org/stable/modules/grid_search.html#tuning-the-hyper-parameters-of-an-estimator>`_.
# In this particular example, we are not implementing the search in order to
# keep the execution time of the example short.

report_rf_reg = evaluate(
    RandomForestRegressor(
        min_samples_leaf=20,
        max_features=0.5,
        random_state=42,
    ),
    X=X,
    y=y,
    splitter=splitter,
)
report_rf_reg.metrics.summarize(data_source="both").frame()

# %%
report_rf_reg.checks.summarize(fast_mode=True)

# %%
# Early stopping
# ==============
#
# For iterative learners, early stopping is another way to limit expressiveness,
# but it does not require a hyperparameter search. Hold out a validation set,
# monitor a metric, and stop when that metric stops improving — further
# iterations are assumed to overfit. See scikit-learn's example on
# `gradient boosting with early stopping
# <https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_early_stopping.html>`_.

report_hgbr_es = evaluate(
    HistGradientBoostingRegressor(
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=40,
        random_state=42,
    ),
    X=X,
    y=y,
    splitter=splitter,
)
report_hgbr_es.metrics.summarize(data_source="both").frame()

# %%
report_hgbr_es.checks.summarize(fast_mode=True)

# %%
# More training data
# ==================
#
# Extra labeled rows help both regimes, but they are especially useful against
# overfitting: the model has fewer opportunities to memorize a small sample.
# Keep one fixed test fold, fit on the original train fold, then refit after
# concatenating the held-out half of the dataset. Use
# :func:`~skore.evaluate` with ``splitter="prefit"`` once the estimator is
# fitted.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_less = HistGradientBoostingRegressor(
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
).fit(X_train, y_train)
report_less = evaluate(model_less, X_test, y_test, splitter="prefit")

model_more = HistGradientBoostingRegressor(
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
).fit(pd.concat([X_train, X_heldout]), pd.concat([y_train, y_heldout]))
report_more = evaluate(model_more, X_test, y_test, splitter="prefit")

compare(
    {"less_training_data": report_less, "more_training_data": report_more}
).metrics.summarize(data_source="test").frame()

# %%
# In production, a `learning curve
# <https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve>`_
# (many refits on growing subsets) shows whether more labels are still worth
# the cost. skore does not wrap that API yet; treat it as a scikit-learn-side
# diagnostic next to these checks.

# %%
# Summary comparison
# ==================
#
# Reading left to right: over-regularized Ridge → enough capacity → too much
# capacity → controlled fit → more data. The same mitigations appear once; only
# the *direction* (add expressiveness vs limit it) changes.

compare(
    {
        "1_ridge_large_alpha": report_underfit,
        "2_ridge": report_ridge,
        "3_hgbr": report_hgbr,
        "4_default_rf": report_rf,
        "5_regularized_rf": report_rf_reg,
        "6_hgbr_early_stopping": report_hgbr_es,
        "7_more_data": report_more,
    }
).metrics.summarize(data_source="test").frame()

# %%
# Conclusion
# ==========
#
# SKD002 and SKD001 are two different consequences of a unsuited expressiveness.
# Relax regularization / add capacity and use informative features until the
# model beats a weak baseline; then regularize, stop early, and add data if
# train scores have a noticeable gap compared to the test scores.
#
# Feature engineering and hyperparameter choices sit in the middle of that path:
# they can rescue an underfit model or, if pushed, create an overfit one. In
# practice you combine several of these levers and re-run the checks until the
# results are satisfying enough.
