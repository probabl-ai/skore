import jedi
import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import make_scorer, mean_squared_error


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
class TestCommon:
    def test_ipython_completion(self, fixture_name, request):
        """Non-regression test for #2386.

        We get no tab completions from IPython if jedi raises an exception, so we
        check here that jedi can produce completions without errors.
        """
        report = request.getfixturevalue(fixture_name)
        if isinstance(report, tuple):
            report = report[0]

        interp = jedi.Interpreter("r.", [{"r": report}])
        interp.complete(line=1, column=2)

    def test_summarize_single_list_equivalence(self, fixture_name, request):
        """Passing a single string is equivalent to passing a list with one element."""
        report = request.getfixturevalue(fixture_name)
        if isinstance(report, tuple):
            report = report[0]

        display_single = report.metrics.summarize(metric="r2")
        display_list = report.metrics.summarize(metric=["r2"])
        assert_frame_equal(display_single.data, display_list.data)

    def test_metrics_repr(self, fixture_name, request):
        """Test that metrics accessor __repr__ returns a string."""
        report = request.getfixturevalue(fixture_name)
        if isinstance(report, tuple):
            report = report[0]

        result = repr(report.metrics)
        assert "metrics" in result.lower()

    def test_metrics_add_scorer(self, fixture_name, request):
        report = request.getfixturevalue(fixture_name)
        if isinstance(report, tuple):
            report = report[0]

        scorer = make_scorer(
            mean_squared_error, greater_is_better=False, response_method="predict"
        )
        report.metrics.add(scorer)

        display = report.metrics.summarize()
        assert "Mean Squared Error" in display.data["metric"].values

    @pytest.mark.parametrize(
        "response_method", ["predict", ["predict", "predict_proba"]]
    )
    def test_metrics_add_callable(self, fixture_name, request, response_method):
        """Check that adding a custom metric by passing the response method works."""
        report = request.getfixturevalue(fixture_name)
        if isinstance(report, tuple):
            report = report[0]

        report.metrics.add(metric=custom_metric, response_method=response_method)
