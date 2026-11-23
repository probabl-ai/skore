"""
.. _example_skrub_data_op_cv:

=======================================================
Using skrub DataOp for cross-validation and evaluation
=======================================================

When a skrub :class:`~skrub.DataOp` defines a cross-validation splitter on
:meth:`~skrub.DataOp.skb.mark_as_X`, :func:`~skore.evaluate` can reuse that
configuration — including ``split_kwargs`` such as ``groups`` — instead of
skore's default 80/20 holdout.

In addition, a :class:`~skrub.DataOp` allows configuring custom metrics,
sample weights, and extra fit arguments that are automatically carried over
into :func:`~skore.evaluate`.

This example builds a grouped cross-validation setup with custom scoring and
fit arguments using skrub and evaluates it with skore.
"""

# %%
# Configure cross-validation on the DataOp
# ========================================
#
# We use the toy products dataset and group products by seller. The goal is to
# assess generalization to new sellers with
# :class:`~sklearn.model_selection.LeaveOneGroupOut`.
import skrub
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneGroupOut

df = skrub.datasets.toy_products()
data = skrub.var("df")
groups = data["seller"]
X = data[["description", "price"]].skb.mark_as_X(
    cv=LeaveOneGroupOut(), split_kwargs={"groups": groups}
)
y = data["category"].skb.mark_as_y()
pred = X.skb.apply(DummyClassifier(), y=y)
learner = pred.skb.make_learner()

# %%
# Evaluate with skore (no explicit splitter)
# ==========================================
#
# Because ``mark_as_X`` was called with an explicit ``cv`` argument, calling
# :func:`~skore.evaluate` without a ``splitter`` returns a
# :class:`~skore.CrossValidationReport` that respects the DataOp grouping.
from skore import evaluate

report = evaluate(learner, data={"df": df})
report

# %%
# There are two sellers, so cross-validation runs in two folds:
len(report.reports_)

# %%
# Inspect aggregated metrics with the same API as other skore reports:
report.metrics.summarize().frame()

# %%
# Configure custom metrics and sample weights
# ===========================================
#
# A :class:`~skrub.DataOp` can attach custom scorers and dynamic parameters
# (such as sample weights) via :meth:`~skrub.DataOp.skb.with_scoring`.
# For instance, we can evaluate a weighted accuracy where each prediction is
# weighted by product price:
sample_weight = X["price"]
pred_weighted = pred.skb.with_scoring(
    "accuracy", kwargs={"sample_weight": sample_weight}, name="weighted_accuracy"
)
weighted_learner = pred_weighted.skb.make_learner()

weighted_report = evaluate(weighted_learner, data={"df": df})
weighted_report.metrics.summarize().frame()

# %%
# Pass additional arguments to estimator fit
# ==========================================
#
# If an estimator requires additional keyword arguments during fitting (for
# example, custom parameters like ``epochs`` or training options), they can be
# passed to ``fit_kwargs`` in :meth:`~skrub.DataOp.skb.apply`:
learner_with_fit_kwargs = X.skb.apply(
    DummyClassifier(), y=y, fit_kwargs={}
).skb.make_learner()
fit_report = evaluate(learner_with_fit_kwargs, data={"df": df})
fit_report

# %%
# Default behavior without an explicit DataOp cv
# ==============================================
#
# If ``mark_as_X`` is called without an explicit ``cv`` argument,
# :func:`~skore.evaluate` still defaults to a single 80/20 holdout and returns
# an :class:`~skore.EstimatorReport`.
simple_learner = skrub.X().skb.apply(DummyClassifier(), y=skrub.y()).skb.make_learner()
holdout_report = evaluate(
    simple_learner,
    data={"X": df[["description", "price"]], "y": df["category"]},
)
holdout_report

# %%
# Explicitly passing a ``splitter`` always overrides the DataOp configuration.
override_report = evaluate(learner, data={"df": df}, splitter=2)
override_report
