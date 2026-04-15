"""
.. _example_custom_metrics:

===========================================================
`EstimatorReport.metrics.add`: Adapt skore to your use-case
===========================================================

By default, :meth:`~skore.EstimatorReport.metrics.summarize` reports a curated
set of metrics for your ML task. In practice you often need domain-specific
scores: a business cost function, a custom fairness measure, an F-beta with a
particular beta, etc.

This example walks through how to register such metrics with
:meth:`~skore.EstimatorReport.metrics.add` so they are computed and displayed
alongside the built-in ones.
"""

# %%
# Setting up a classification problem
# ===================================

# %%
import skore
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)

# %%
# We create an :class:`~skore.EstimatorReport` through :func:`~skore.evaluate`
# using a simple train/test split. ``pos_label=1`` marks the *malignant* class
# as the positive class.

# %%
report = skore.evaluate(
    LogisticRegression(max_iter=10_000), X, y, pos_label=1, splitter=0.2
)

# %%
# Let's look at the default metrics:
report.metrics.summarize().frame()

# %%
# Adding a plain callable
# =======================
#
# Any function with the signature ``(y_true, y_pred, **kwargs) -> float`` can be
# registered with :meth:`~skore.EstimatorReport.metrics.add`. The function name
# is used as the metric name by default.


# %%
def specificity(y_true, y_pred):
    """Proportion of true negatives among actual negatives."""
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tn / (tn + fp)


report.metrics.add(specificity, greater_is_better=True)

# %%
report.metrics.summarize().frame()

# %%
# ``specificity`` now appears alongside the built-in metrics.

# %%
# Passing extra keyword arguments
# ================================
#
# If your metric needs extra data at scoring time (e.g. sample-level amounts,
# a cost matrix, ...), pass them as keyword arguments to
# :meth:`~skore.EstimatorReport.metrics.add`. They will be forwarded to the
# metric function when it is computed.

# %%


def misclassification_cost(y_true, y_pred, cost_fp, cost_fn):
    """Total cost of misclassifications, weighted by error type."""
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return cost_fp * fp + cost_fn * fn


report.metrics.add(
    misclassification_cost,
    greater_is_better=False,
    cost_fp=1.0,
    cost_fn=10.0,
)

report.metrics.summarize().frame()

# %%
# Adding an sklearn scorer
# ========================
#
# If you already have a :func:`~sklearn.metrics.make_scorer` object, you can
# register it directly. The ``response_method`` and ``greater_is_better``
# metadata are extracted from the scorer automatically.

# %%
from sklearn.metrics import fbeta_score, make_scorer

f2_scorer = make_scorer(fbeta_score, beta=2, response_method="predict", pos_label=1)
report.metrics.add(f2_scorer, name="f2")

report.metrics.summarize().frame()

# %%
# Cherry-picking metrics to display
# ==================================
#
# Once registered, custom metrics can be selected by name in
# :meth:`~skore.EstimatorReport.metrics.summarize`:

# %%
report.metrics.summarize(
    metric=["specificity", "f2", "misclassification_cost"],
).frame()

# %%
# Selecting ``data_source="both"`` lets you compare train vs. test in one call:

# %%
report.metrics.summarize(metric=["specificity", "f2"], data_source="both").frame()

# %%
# Using a different response method
# ==================================
#
# By default, callables receive the output of ``estimator.predict(X)``. If your
# metric needs probabilities instead, set ``response_method="predict_proba"``.

# %%
import numpy as np


def mean_confidence(y_true, y_proba):
    """Average predicted probability assigned to the true class."""
    return np.where(y_true == 1, y_proba, 1 - y_proba).mean()


report.metrics.add(
    mean_confidence, response_method="predict_proba", greater_is_better=True
)

report.metrics.summarize(metric="mean_confidence").frame()
