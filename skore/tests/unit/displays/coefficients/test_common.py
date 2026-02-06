import pytest

from skore import CoefficientsDisplay


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports",
        "cross_validation_reports",
        "comparison_estimator_reports",
        "comparison_cross_validation_reports",
    ],
)
@pytest.mark.parametrize(
    "task",
    [
        "binary_classification",
        "multiclass_classification",
        "regression",
        "multioutput_regression",
    ],
)
class TestCoefficientsDisplay:
    def test_class_attributes(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.coefficients()
        assert isinstance(display, CoefficientsDisplay)
        assert hasattr(display, "coefficients")
        assert hasattr(display, "report_type")

        display.plot()
        assert hasattr(display, "facet_")
        assert hasattr(display, "figure_")
        assert hasattr(display, "ax_")

    def test_frame_structure(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.coefficients()
        frame = display.frame(sorting_order=None)

        expected = {"feature", "coefficients"}
        if "cross_validation" in fixture_prefix:
            expected.add("split")
        if "comparison" in fixture_prefix:
            expected.add("estimator")
        if task == "multiclass_classification":
            expected.add("label")
        if task == "multioutput_regression":
            expected.add("output")
        assert set(frame.columns) == expected

    def test_internal_data_structure(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.coefficients()
        expected = {
            "estimator",
            "split",
            "feature",
            "label",
            "output",
            "coefficients",
        }
        assert set(display.coefficients.columns) == expected

    def test_plot_structure(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        _, ax = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        if hasattr(ax, "flatten"):
            ax = ax.flatten()[0]
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""

    def test_title(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.coefficients()
        figure, _ = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        title = figure.get_suptitle()
        assert "Coefficients" in title
        if "comparison" not in fixture_prefix:
            estimator_name = display.coefficients["estimator"].iloc[0]
            assert estimator_name in title

    def test_kwargs(pyplot, fixture_prefix, task, request):
        """Check that custom `barplot_kwargs` are applied to the plots."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.coefficients()
        figure, _ = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        assert figure.get_figheight() == 6

        if "estimator_reports" in fixture_prefix:
            display.set_style(barplot_kwargs={"height": 8}).plot()
        else:
            display.set_style(stripplot_kwargs={"height": 8}).plot()
        assert display.figure_.get_figheight() == 8
