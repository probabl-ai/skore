"""Tests for the metrics registry that are specific to ``ComparisonReport``."""

from sklearn.metrics import make_scorer, mean_squared_error


class TestSummarizeIntegration:
    def test_summarize_explicit_custom_metric(
        self, linear_regression_comparison_report
    ):
        """``summarize`` exposes the per-report ``estimator_name`` column."""
        report = linear_regression_comparison_report
        report.metrics.add(
            make_scorer(
                mean_squared_error,
                greater_is_better=False,
                response_method="predict",
            )
        )
        display = report.metrics.summarize(metric="mean_squared_error")
        assert set(display.data["estimator_name"]) == {"estimator_1", "estimator_2"}
        assert set(display.data["metric_verbose_name"]) == {"Mean Squared Error"}
