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
An untuned gradient boosting regressor triggers SKD001; tuning hyperparameters
then narrows the train/test gap. For the feature-engineering section, we
deliberately withhold one moderately useful predictor, inject a spurious
``house_id``, then drop the bad column and restore the useful one.
"""

# %%
# Load the California housing dataset
# ===================================
#
# Each row describes a block in California. The target is ``MedHouseVal``
# in \$100k units; we multiply by 100 so errors read naturally in k$.
# We keep a 5,000-row subsample so this example stays short.

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

X_full = X_full.reset_index(drop=True)
y = pd.Series(y, name="MedHouseVal_k$").reset_index(drop=True)

house_age = X_full["HouseAge"]
X = X_full.drop(columns=["HouseAge"])

# %%
# We use :class:`~skrub.TableReport` to visualize the feature matrix and the target.

from skrub import TableReport

TableReport(X)

# %%

TableReport(y)

# %%
# A single random train-test split is used for the following cells.

from skore import TrainTestSplit

splitter = TrainTestSplit(random_state=42)

# %%
# Trigger SKD001 — untuned gradient boosting
# ===========================================
#
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor` with default settings
# has enough capacity to overfit the training data on this table. Compare train and
# test rows in the metrics table, then run checks to see if SKD001 is triggered.

from sklearn.ensemble import HistGradientBoostingRegressor
from skore import evaluate


def untuned_hgbr():
    return HistGradientBoostingRegressor(random_state=42)


estimator = untuned_hgbr()

report = evaluate(
    estimator,
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
# Lower ``max_depth``, larger ``min_samples_leaf``, and a slower
# ``learning_rate`` can reduce memorization.

report_regularized = evaluate(
    HistGradientBoostingRegressor(
        max_depth=3,
        min_samples_leaf=50,
        learning_rate=0.05,
        max_iter=200,
        random_state=42,
    ),
    X=X,
    y=y,
    splitter=splitter,
)

report_regularized.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_regularized.checks.summarize()

# SKD001 clears once the train/test gap narrows.
# %%
# Use more training data
# ======================
#
# More rows give the model more examples of stable patterns. We hold
# out one fixed test set, fit the same untuned estimator on two training subsets
# of different size, and compare test metrics on identical test holdouts.

from skore import EstimatorReport, compare

X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)  # same 20% hold-out as ``splitter``

X_train_small, _, y_train_small, _ = train_test_split(
    X_train_pool,
    y_train_pool,
    train_size=0.5,
    random_state=42,
)  # half of the training pool

report_less_data = EstimatorReport(
    untuned_hgbr(),
    X_train=X_train_small,
    y_train=y_train_small,
    X_test=X_test,
    y_test=y_test,
)

report_more_data = EstimatorReport(
    untuned_hgbr(),
    X_train=X_train_pool,
    y_train=y_train_pool,
    X_test=X_test,
    y_test=y_test,
)

# %%
# The gap between test and train metrics is typically reduced when the model sees more
# training rows.

report_more_data.metrics.summarize(data_source="both").frame(favorability=True)

# %%
report_more_data.checks.summarize()

# Even though the gap is reduced, SKD001 is still triggered.

# %%
# Improve feature engineering
# ===========================
#
# Feature engineering can be done here in two ways : dropping columns that
# invite memorization and adding  columns that carry real signal. We first
# inject ``house_id`` — a unique numeric identifier with no meaning for house
# prices — then remove it and add ``HouseAge``, the column we set aside at load
# time which is an important feature for predicting house prices.

import numpy as np

n_samples = len(X)
house_id = pd.Series(np.arange(n_samples), index=X.index, name="house_id")
X_spurious = X.assign(house_id=house_id)

report_spurious = evaluate(
    untuned_hgbr(),
    X=X_spurious,
    y=y,
    splitter=splitter,
)

report_spurious.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# ``house_id`` gives the booster a unique value for each row, and the gap between
# test and train metrics is increased.

report_spurious.checks.summarize()

# %%
X_engineered = X_spurious.drop(columns=["house_id"]).assign(HouseAge=house_age)

report_fe = evaluate(
    untuned_hgbr(),
    X=X_engineered,
    y=y,
    splitter=splitter,
)

report_fe.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Removing ``house_id`` and restoring ``HouseAge`` reduces the gap between test and train
# metrics relative to the spurious table. SKD001 may still fire because the booster remains
# untuned; feature quality and regularization should be done together in practice.

report_fe.checks.summarize()

# %%
# Compare mitigations
# ===================
#
# :func:`~skore.compare` lines up reports on a shared test set. Each compare
# below isolates one lever: hyperparameters, training-set size, or features.
# Mixing them in a single compare would conflate causes.

comparison = compare(
    {
        "default_hgbr": report,
        "tuned_hgbr": report_regularized,
    }
)
comparison.metrics.summarize(data_source="both").frame(favorability=True)

# %%
comparison_data = compare(
    {
        "less_training_data": report_less_data,
        "more_training_data": report_more_data,
    }
)
comparison_data.metrics.summarize(data_source="both").frame(favorability=True)

# %%
comparison_fe = compare(
    {
        "baseline": report,
        "with_house_id": report_spurious,
        "engineered": report_fe,
    }
)
comparison_fe.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Conclusion
# ==========
#
# SKD001 warns against a common pitfall of data science models which is memorization
# instead of generalization. Here, an untuned gradient boosting regressor on California
# housing showed a wide train/test gap; tuning depth, leaf size, and learning
# rate narrowed it. More training rows improved the gap between test and train scores on
# the same hold-out set; dropping ``house_id`` and adding back ``HouseAge`` helped too. Combine
# regularization, more data, and thoughtful features in practice until SKD001 is cleared.
