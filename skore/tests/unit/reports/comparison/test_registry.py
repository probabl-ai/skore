"""Tests for the metrics registry that are specific to ``ComparisonReport``."""

from sklearn.metrics import make_scorer, mean_squared_error


def test_summarize_explicit_custom_metric(linear_regression_comparison_report):
    """``summarize`` exposes the per-report ``estimator`` column."""
    report = linear_regression_comparison_report
    report.metrics.add(
        make_scorer(
            mean_squared_error,
            greater_is_better=False,
            response_method="predict",
        )
    )
    display = report.metrics.summarize(metric="mean_squared_error")
    assert set(display.summary["estimator"]) == {"estimator_1", "estimator_2"}
    assert set(display.summary["verbose_name"]) == {"Mean Squared Error"}
