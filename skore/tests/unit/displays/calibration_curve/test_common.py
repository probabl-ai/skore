import pytest

from skore import CalibrationDisplay


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports",
    ],
)
@pytest.mark.parametrize("task", ["binary_classification"])
class TestCalibrationDisplay:
    def test_class_attributes(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.calibration_curve()
        assert isinstance(display, CalibrationDisplay)
        assert hasattr(display.calibration_report, "predicted_probability")
        assert hasattr(display.calibration_report, "fraction_of_positives")
        assert hasattr(display.calibration_report, "data_source")
        assert hasattr(display, "report_pos_label")
        fig = display.plot()
        assert fig is not None
        assert len(fig.axes) >= 1

    def test_frame_structure(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.calibration_curve()
        frame = display.frame()
        expected = {
            "predicted_probability",
            "fraction_of_positives",
            "data_source",
            "label",
        }
        assert set(frame.columns) == expected

    def test_internal_data_structure(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.calibration_curve()
        expected = {
            "predicted_probability",
            "fraction_of_positives",
            "data_source",
            "label",
        }
        assert set(display.frame().columns) == expected

    def test_plot_structure(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        _, axes = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        ax = axes[0]
        assert ax.get_xlabel() == "Mean predicted probability"
        assert ax.get_ylabel() == "Fraction of positives"

    def test_title(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.calibration_curve()
        figure, _ = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        title = figure.get_suptitle()
        assert "Calibration Curve" in title
        if "comparison" not in fixture_prefix:
            estimator_name = display.calibration_report["estimator"].iloc[0]
            assert estimator_name in title

    @pytest.mark.xfail(
        reason="Currently, the calibration curve display does not support kwargs. "
        + "This test should be updated once that functionality is added."
    )
    def test_kwargs(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.calibration_curve()
        figure, _ = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        assert figure.get_figheight() == 6
        if "estimator" in fixture_prefix:
            display.set_style(barplot_kwargs={"height": 8})
            fig = display.plot()
        else:  # "cross_validation"
            display.set_style(stripplot_kwargs={"height": 8})
            fig = display.plot()
        assert fig.get_figheight() == 8
