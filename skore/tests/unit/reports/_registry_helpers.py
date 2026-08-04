"""Metric callables shared by the metrics registry test modules."""


def business_loss_metric(y_true, y_pred, *, cost_fp, cost_fn):
    """Custom ``(y_true, y_pred)`` metric: weighted cost of false positives/negatives.

    Its first argument is ``y_true``, which is what makes it a valid input for
    :func:`sklearn.metrics.make_scorer` and an invalid one for ``Metric.new``.
    """
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return fp * cost_fp + fn * cost_fn


def business_loss_scorer(estimator, X, y, cost_fp, cost_fn):
    """Custom ``(estimator, X, y)`` scorer with required kwargs."""
    return business_loss_metric(
        y, estimator.predict(X), cost_fp=cost_fp, cost_fn=cost_fn
    )
