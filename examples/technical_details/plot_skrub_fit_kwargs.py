"""
.. _example_skrub_fit_kwargs:

===================================================
Passing additional fit parameters via skrub Learner
===================================================

Some predictive models or wrappers require additional arguments to their
:meth:`~sklearn.base.BaseEstimator.fit` method (such as ``fit_kwargs``,
hyperparameters like ``epochs``, or custom training options). While
:class:`~skore.EstimatorReport` and :class:`~skore.CrossValidationReport`
accept features ``X`` and target ``y``, they do not accept arbitrary
estimator-specific keyword arguments directly.

Instead, you can use :mod:`skrub` to package the estimator and its keyword
arguments into a :class:`~skrub.SkrubLearner`, then evaluate it seamlessly with
:func:`~skore.evaluate`.

This example demonstrates how to pass additional arguments to an estimator's
``fit`` method using skrub and evaluate it with skore.
"""

# %%
# A model requiring fit arguments
# ===============================
#
# Consider a custom estimator or a deep learning wrapper whose ``fit`` method
# expects an extra keyword argument, such as ``epochs``:
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge


class EpochAwareRidge(RegressorMixin, BaseEstimator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, epochs=1):
        # In practice, epochs may control iterations in a deep learning training loop.
        # Here we scale regularization based on epochs as an illustrative example.
        self.epochs_ = epochs
        self.estimator_ = Ridge(alpha=self.alpha / epochs)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


# %%
# Generate synthetic regression data:
X, y = make_regression(n_samples=100, n_features=5, random_state=42)

# %%
# Package the estimator with fit_kwargs using skrub
# =================================================
#
# We use :meth:`~skrub.DataOp.skb.apply` to bind the estimator and pass the
# required ``fit_kwargs`` (here, ``epochs=10``). We then build a learner
# with :meth:`~skrub.DataOp.skb.make_learner`:
import skrub

learner = (
    skrub.X()
    .skb.apply(EpochAwareRidge(alpha=1.0), y=skrub.y(), fit_kwargs={"epochs": 10})
    .skb.make_learner()
)

# %%
# Evaluate with a single train-test split
# =======================================
#
# Pass the learner to :func:`~skore.evaluate` along with a dictionary of data
# bindings. Because the default splitter is a single holdout split (80/20),
# this returns an :class:`~skore.EstimatorReport`:
from skore import evaluate

report = evaluate(learner, data={"X": X, "y": y}, splitter=0.2)
report

# %%
# We can inspect the summary of performance metrics:
report.metrics.summarize().frame()

# %%
# Evaluate with cross-validation
# ==============================
#
# The same skrub learner can be evaluated using cross-validation by specifying
# an integer number of folds or a CV splitter object:
cv_report = evaluate(learner, data={"X": X, "y": y}, splitter=3)
cv_report

# %%
# Inspect the aggregated cross-validation metrics across splits:
cv_report.metrics.summarize().frame()

