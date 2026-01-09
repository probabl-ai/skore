import numpy as np
import pytest


def custom_metric(y_true, y_pred, threshold=0.5):
    residuals = y_true - y_pred
    return np.mean(np.where(residuals < threshold, residuals, 1))


@pytest.mark.parametrize(
    "fixture_name",
    [
        "estimator_reports_regression",
        "cross_validation_reports_regression",
        "comparison_estimator_reports_regression",
        "comparison_cross_validation_reports_regression",
    ],
)
class TestCustomMetricSummarize:
    @pytest.mark.parametrize(
        "response_method", ["predict", ["predict", "predict_proba"]]
    )
    def test_works(self, fixture_name, request, response_method):
        report = request.getfixturevalue(fixture_name)
        if isinstance(report, tuple):
            report = report[0]

        report.metrics.summarize(
            metric=custom_metric, response_method=response_method
        ).frame()

    def test_no_response_method(self, fixture_name, request):
        report = request.getfixturevalue(fixture_name)
        if isinstance(report, tuple):
            report = report[0]

        err_msg = "response_method is required when the metric is a callable"
        with pytest.raises(ValueError, match=err_msg):
            report.metrics.summarize(metric=custom_metric).frame()
