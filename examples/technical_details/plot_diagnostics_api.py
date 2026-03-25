"""
.. _example_diagnostics_api:

===================
The diagnostics API
===================

`skore` can automatically detect common modeling pitfalls such as overfitting
and underfitting. This example walks through the diagnostics API: how to
trigger diagnostics, how to read the results, and how to mute specific checks.

We use a purely non-linear regression target and deliberately pick models that
fail in known ways:

- a **linear model** that cannot capture the non-linearity → underfitting,
- a **single deep decision tree** that memorizes the training set perfectly
  and fails to generalize → overfitting.
"""

# %%
# Setup
# =====
#
# The target is a product of trigonometric functions of the first two features:
# completely invisible to a linear model, yet easy to memorize for an
# unconstrained tree.

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(42)
n_samples = 500
X = rng.uniform(0, 1, (n_samples, 5))
y = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1]) + rng.normal(
    0, 0.1, n_samples
)

linear = LinearRegression()
deep_tree = DecisionTreeRegressor(random_state=42)

# %%
# Calling :meth:`~skore.EstimatorReport.diagnose` explicitly
# ==========================================================
#
# Every report exposes a :meth:`~skore.EstimatorReport.diagnose` method.
# By default, diagnostics are **not** displayed at creation time — you call
# :meth:`~skore.EstimatorReport.diagnose` yourself.

from skore import evaluate

linear_report = evaluate(linear, X, y)
linear_report.diagnose()

# %%
linear_report.metrics.summarize(data_source="both").frame()

# %%
# The linear model is flagged for underfitting: its scores are on par between
# train and test, and not significantly better than a dummy baseline.

tree_report = evaluate(deep_tree, X, y)
tree_report.diagnose()

# %%
tree_report.metrics.summarize(data_source="both").frame()

# %%
# The deep tree is flagged for overfitting: it achieves a perfect score on
# train but degrades on test.

# %%
# Showing diagnostics at creation time
# ====================================
#
# Pass ``diagnose=True`` to :func:`~skore.evaluate` (or to any report
# constructor) to run and display diagnostics immediately.

tree_report = evaluate(deep_tree, X, y, diagnose=True)

# %%
# You can also turn this on globally with :obj:`~skore.configuration` as a
# context manager:

import skore

with skore.configuration(diagnose=True):
    tree_report = evaluate(deep_tree, X, y)

# %%
# Ignoring specific checks
# ========================
#
# Each diagnostic has a stable code (e.g. ``SKD001``, ``SKD002``). You can
# mute individual checks per call:

tree_report.diagnose(ignore=["SKD001"])

# %%
# Or globally, so that every subsequent :meth:`~skore.EstimatorReport.diagnose` call
# skips them:

with skore.configuration(ignore_diagnostics=["SKD001"]):
    diagnosis = tree_report.diagnose()
diagnosis

# %%
# Diagnostics on a :class:`~skore.CrossValidationReport`
# =====================================================
#
# When ``splitter`` is an integer, :func:`~skore.evaluate` returns a
# :class:`~skore.CrossValidationReport`. Diagnostics aggregate across folds.

cv_report = evaluate(deep_tree, X, y, splitter=5)
cv_report.diagnose()

# %%
# Diagnostics on a :class:`~skore.ComparisonReport`
# ================================================
#
# Passing a list of estimators returns a :class:`~skore.ComparisonReport`.
# Diagnostics are grouped by sub-report.

comparison_report = evaluate([linear, deep_tree], X, y)
comparison_report.diagnose()
