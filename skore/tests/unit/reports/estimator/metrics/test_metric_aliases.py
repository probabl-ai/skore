"""Tests for _MetricsAccessor._METRIC_ALIASES.

Verify that user-friendly metric names (without the ``neg_`` prefix) resolve
to the same scorer and produce the same scores as their ``neg_``-prefixed
sklearn equivalents.
"""

import numpy as np
import pytest

from skore import EstimatorReport
from skore._sklearn._estimator.metrics_accessor import _MetricsAccessor

# ---------------------------------------------------------------------------
# _resolve_metric_alias unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("mean_squared_error", "neg_mean_squared_error"),
        ("mean_absolute_error", "neg_mean_absolute_error"),
        ("mean_absolute_percentage_error", "neg_mean_absolute_percentage_error"),
        ("median_absolute_error", "neg_median_absolute_error"),
        ("mean_squared_log_error", "neg_mean_squared_log_error"),
        ("root_mean_squared_error", "neg_root_mean_squared_error"),
        ("root_mean_squared_log_error", "neg_root_mean_squared_log_error"),
        ("mean_poisson_deviance", "neg_mean_poisson_deviance"),
        ("mean_gamma_deviance", "neg_mean_gamma_deviance"),
        ("max_error", "neg_max_error"),
        ("negative_likelihood_ratio", "neg_negative_likelihood_ratio"),
    ],
)
def test_resolve_metric_alias_known(alias, expected):
    """Known aliases are resolved to their neg_ equivalents."""
    assert _MetricsAccessor._resolve_metric_alias(alias) == expected


@pytest.mark.parametrize("metric", ["accuracy", "f1", "r2", "neg_mean_squared_error"])
def test_resolve_metric_alias_passthrough(metric):
    """Non-aliased strings (including neg_ prefixed ones) pass through unchanged."""
    assert _MetricsAccessor._resolve_metric_alias(metric) == metric


def test_resolve_metric_alias_non_string():
    """Non-string metrics (e.g. callables) are returned unchanged."""
    func = lambda y_true, y_pred: 0.0  # noqa: E731
    assert _MetricsAccessor._resolve_metric_alias(func) is func


# ---------------------------------------------------------------------------
# Integration tests via summarize()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric_without_neg, metric_with_neg",
    [
        ("mean_squared_error", "neg_mean_squared_error"),
        ("mean_absolute_error", "neg_mean_absolute_error"),
        ("root_mean_squared_error", "neg_root_mean_squared_error"),
    ],
)
def test_summarize_without_neg_prefix(
    linear_regression_with_test, metric_without_neg, metric_with_neg
):
    """summarize(metric='mean_squared_error') produces the same result as
    summarize(metric='neg_mean_squared_error')."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display_without = report.metrics.summarize(metric=[metric_without_neg])
    display_with = report.metrics.summarize(metric=[metric_with_neg])

    assert set(display_without.data["metric"]) == set(display_with.data["metric"])
    np.testing.assert_allclose(
        display_without.data["score"].values, display_with.data["score"].values
    )


def test_summarize_alias_in_dict(linear_regression_with_test):
    """Aliases also work when metrics are passed as a dict."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(
        metric={"My MSE": "mean_squared_error", "My MAE": "mean_absolute_error"}
    )

    assert set(display.data["metric"]) == {"My MSE", "My MAE"}
    assert all(display.data["score"] >= 0)


def test_summarize_alias_single_string(linear_regression_with_test):
    """A single alias string (not in a list) also resolves correctly."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric="mean_squared_error")

    assert len(display.data) == 1
    assert display.data["score"].values[0] >= 0
