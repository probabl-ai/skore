"""
.. _example_skrub_data_op_cv:

===================================
Using skrub DataOp cross-validation
===================================

When a skrub :class:`~skrub.DataOp` defines a cross-validation splitter on
:meth:`~skrub.DataOp.skb.mark_as_X`, :func:`~skore.evaluate` can reuse that
configuration — including ``split_kwargs`` such as ``groups`` — instead of
skore's default 80/20 holdout.

This example builds a small grouped cross-validation setup with skrub and
evaluates it with skore.
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
data = skrub.var("df", df)
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
