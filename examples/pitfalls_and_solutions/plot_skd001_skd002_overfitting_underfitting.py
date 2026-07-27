"""
.. _example_skd001_skd002_overfitting_underfitting:

SKD001 & SKD002 - Overfitting and underfitting
==============================================

:ref:`SKD002 <skd002-underfitting>` and :ref:`SKD001 <skd001-overfitting>`
describe opposite ends of the same problem: model *expressiveness*.

- Too little expressiveness: train and test scores are close to those of a
  model that guesses randomly (often called a dummy model) (SKD002).
- Too much expressiveness: train scores pull far ahead of test scores
  (SKD001).

A performant model learns from the training data without memorizing its
specificities, so that learned knowledge can be generalized to new data. This
notebook walks the underfitting to overfitting path while showing different
mitigations techniques:

- tweaking model capacity (model family / complexity)
- adding and removing features
- tuning regularization
- using early stopping
- using more data

We use California housing (median house value in kUSD). Half of the rows feed
the main walkthrough; the other half is reserved to show the effect of adding
training data.
"""

# %%
# Load the California housing dataset
# ===================================
#
# Each row is a census block group. The target ``MedHouseVal`` is in 100k USD
# units; we multiply by 100 so errors read in kUSD.

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

y = pd.Series(y, name="MedHouseVal_kUSD")
y_heldout = pd.Series(y_heldout, name="MedHouseVal_kUSD")

# %%
from skrub import TableReport

TableReport(X)

# %%
TableReport(y)

# %%
# We use the same split for every model comparison, with a fixed seed, so that
# we are sure that we are comparing models on the same test data.

from skore import TrainTestSplit

splitter = TrainTestSplit(test_size=0.2, random_state=42)

# %%
# Underfitting: SKD002 fires
# ==========================
#
# We start with a linear model that is overly regularized.
# :class:`~sklearn.linear_model.Ridge` with a very large ``alpha`` shrinks
# coefficients toward zero, so predictions stay close to a dummy baseline.

import skore
from sklearn.linear_model import Ridge

report_underfit = skore.evaluate(
    Ridge(alpha=1e6),
    X=X,
    y=y,
    splitter=splitter,
)
report_underfit.metrics.summarize(data_source="both").frame()

# %%
# For a model that learns so little, SKD002 fires.

report_underfit.checks.summarize(fast_mode=True)

# %%
# Increase model expressiveness
# =============================
#
# Moving away from underfitting means giving the model enough expressiveness to
# use the inputs. Dropping to a :class:`~sklearn.linear_model.Ridge` with a much
# smaller ``alpha`` already learns useful weights on each feature and is enough
# to clear SKD002 on this table.

report_ridge = skore.evaluate(Ridge(alpha=1.0), X=X, y=y, splitter=splitter)
report_ridge.metrics.summarize(data_source="both").frame()

# %%
report_ridge.checks.summarize(fast_mode=True)

# %%
# A :class:`~sklearn.ensemble.HistGradientBoostingRegressor` adds nonlinear
# capacity and can lift test scores further. Watch the train/test gap as you do
# this: the same method that cures underfitting is the one that creates
# overfitting.

from sklearn.ensemble import HistGradientBoostingRegressor

report_hgbr = skore.evaluate(
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
# Better features help when the model family is fine but the representation
# is weak (still an underfitting problem). Here, ``AveRooms / AveOccup`` is a
# rooms-per-person signal that a linear model can use more easily than the raw
# counts.
#
# The same step has an overfitting side effect: every new feature also increases
# expressiveness. Mild engineering can close an underfit gap; aggressive
# expansions (very high-degree interactions, huge one-hot spaces) can reopen an
# overfit gap. Think of features as capacity you add to the inputs, not only
# to the estimator.

X_fe = X.assign(RoomsPerPerson=X["AveRooms"] / X["AveOccup"].clip(lower=0.1))

report_ridge_fe = skore.evaluate(Ridge(alpha=1.0), X=X_fe, y=y, splitter=splitter)
report_ridge_fe.metrics.summarize(data_source="both").frame()

# %%
# The test score rises while the train score stays close: we reduced underfitting
# without creating overfitting by adding a signal-rich feature.

report_ridge_fe.checks.summarize(fast_mode=True)

# %%
# Overfitting: SKD001 fires
# =========================
#
# To see the other end of the spectrum on the same split, fit a default
# :class:`~sklearn.ensemble.RandomForestRegressor`. Unrestricted leaves can
# memorize training data really well so train metrics look excellent, test metrics
# lag, and SKD001 flags the gap.

from sklearn.ensemble import RandomForestRegressor

report_rf = skore.evaluate(
    RandomForestRegressor(random_state=42),
    X=X,
    y=y,
    splitter=splitter,
)
report_rf.metrics.summarize(data_source="both").frame()

# %%
report_rf.checks.summarize(fast_mode=True)

# %%
# Feature selection
# =================
#
# Feature selection is the other direction from engineering: drop columns that
# do not carry enough signal so the model has less room to memorize noise. We
# run :class:`~sklearn.feature_selection.RFECV` with a random forest and
# rank features via :func:`~sklearn.inspection.permutation_importance` through
# ``importance_getter``.
#
# RFECV calls ``importance_getter(model)`` only, so we:
#
# - split a held-out validation set ourselves for permutation importance (not
#   the skore evaluation test fold, to avoid leakage),
# - close over that hold-out in a ``lambda`` that still matches the one-argument
#   getter signature,
# - return ``importances_mean`` from the permutation-importance result.

import inspect

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

X_sel = X.sample(3_000, random_state=42)
y_sel = y.loc[X_sel.index]
X_fs, X_val, y_fs, y_val = train_test_split(
    X_sel, y_sel, test_size=0.2, random_state=42
)
X_val_np = np.asarray(X_val)
y_val_np = np.asarray(y_val)


def permutation_importance_getter(model, X_val, y_val, random_state=42):
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)
    if X_val.shape[1] != model.n_features_in_:
        features = None
        frame = inspect.currentframe()
        try:
            caller = frame.f_back
            while caller is not None:
                local_ns = caller.f_locals
                if "features" in local_ns and "support_" in local_ns:
                    features = local_ns["features"]
                    break
                caller = caller.f_back
        finally:
            del frame
        if features is None:
            raise RuntimeError(
                "Could not recover active RFECV feature indices for the hold-out set."
            )
        X_val = X_val[:, features]
    return permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=1,
        random_state=random_state,
        n_jobs=None,
    ).importances_mean


rf_regressor = RandomForestRegressor(
    n_estimators=20,
    max_depth=10,
    random_state=42,
    n_jobs=1,
)

rfecv = RFECV(
    estimator=rf_regressor,
    step=1,
    cv=KFold(2, shuffle=True, random_state=42),
    scoring="r2",
    n_jobs=None,
    importance_getter=lambda model: permutation_importance_getter(
        model, X_val_np, y_val_np, random_state=42
    ),
)

report_rf_selected = skore.evaluate(
    rfecv,
    X=X_fs,
    y=y_fs,
    splitter=splitter,
)

# Same subsample and splitter as RFECV so ``skore.compare`` shares test targets.
report_rf_fs = skore.evaluate(
    RandomForestRegressor(random_state=42),
    X=X_fs,
    y=y_fs,
    splitter=splitter,
)

skore.compare(
    {
        "rf_all_features": report_rf_fs,
        "rf_after_rfecv": report_rf_selected,
    }
).metrics.summarize(data_source="both").frame()

# %%
# After RFECV, test metrics move up a little and the train/test gap narrows:
# the forest focuses on the stronger housing features. Selection helps, but
# further capacity control is still useful if SKD001 remains.

report_rf_selected.checks.summarize(fast_mode=True)

# %%
selected_columns = list(X_fs.columns[report_rf_selected.estimator_.support_])
dropped_columns = list(X_fs.columns[~report_rf_selected.estimator_.support_])
selected_columns, dropped_columns

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

model_rf_reg = RandomForestRegressor(
    min_samples_leaf=20,
    max_features=0.5,
    random_state=42,
)
report_rf_reg = skore.evaluate(
    model_rf_reg,
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
# For iterative learners, early stopping is another way to limit expressiveness
# that does not require a hyperparameter search. Hold out a validation set,
# monitor a metric, and stop when that metric stops improving. Further
# iterations are assumed to overfit. Either pass a validation set to ``.fit``
# or use the ``validation_fraction`` parameter. See scikit-learn's example on
# `gradient boosting with early stopping
# <https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_early_stopping.html>`_.

model_hgbr_es = HistGradientBoostingRegressor(
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    max_iter=40,
    random_state=42,
)
report_hgbr_es = skore.evaluate(
    model_hgbr_es,
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
# fitted. A side-effect is that fit time is unavailable (skore did not time
# ``.fit``), so that metric appears as NaN in the reported tables.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_less = HistGradientBoostingRegressor(
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
).fit(X_train, y_train)
report_less = skore.evaluate(model_less, X_test, y_test, splitter="prefit")

model_more = HistGradientBoostingRegressor(
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
).fit(pd.concat([X_train, X_heldout]), pd.concat([y_train, y_heldout]))
report_more = skore.evaluate(model_more, X_test, y_test, splitter="prefit")

skore.compare(
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

skore.compare(
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
# model beats a weak baseline; then select features, regularize, stop early,
# and add data if train scores have a noticeable gap compared to the test
# scores.
#
# Feature engineering and feature selection are opposite middle steps:
# adding informative columns can rescue an underfit model; dropping weaker
# ones with RFECV can shrink an overfit gap. Hyperparameter choices play a
# similar dual role. In practice you combine several of these levers and
# re-run the checks until the results are satisfying enough.
