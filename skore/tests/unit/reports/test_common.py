import jedi
import pytest
from pandas.testing import assert_frame_equal


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
