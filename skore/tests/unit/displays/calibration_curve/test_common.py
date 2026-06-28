import pytest

from skore import CalibrationDisplay


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports",
        "cross_validation_reports",
    ],
)
@pytest.mark.parametrize("task", ["binary_classification", "multiclass_classification"])
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
        if "cross_validation" in fixture_prefix:
            expected = {
                "predicted_probability_mean",
                "predicted_probability_std",
                "fraction_of_positives_mean",
                "fraction_of_positives_std",
                "data_source",
                "label",
            }
        else:
            expected = {
                "predicted_probability",
                "fraction_of_positives",
                "data_source",
                "label",
            }
        assert set(frame.columns) == expected

    def test_frame_aggregate(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.calibration_curve()

        if "cross_validation" not in fixture_prefix:
            # aggregate is a no-op for non-CV report types
            assert set(display.frame().columns) == set(
                display.frame(aggregate=None).columns
            )
        else:
            # aggregate=None returns one curve per split with raw column names
            frame_none = display.frame(aggregate=None)
            assert "split" in frame_none.columns
            assert "predicted_probability" in frame_none.columns
            assert "fraction_of_positives" in frame_none.columns
            assert "predicted_probability_mean" not in frame_none.columns

            # default aggregate=("mean", "std") collapses splits
            frame_agg = display.frame()
            assert "split" not in frame_agg.columns
            assert "predicted_probability_mean" in frame_agg.columns
            assert "fraction_of_positives_std" in frame_agg.columns
            assert "predicted_probability" not in frame_agg.columns

    def test_internal_data_structure(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.calibration_curve()
        expected = {
            "estimator",
            "data_source",
            "split",
            "label",
            "predicted_probability",
            "fraction_of_positives",
        }
        assert set(display.calibration_report.columns) == expected

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

    def test_kwargs(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.calibration_curve()
        fig = display.plot()
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Mean predicted probability"
        assert ax.get_ylabel() == "Fraction of positives"
        assert tuple(ax.get_xlim()) == (0.0, 1.0)
        assert tuple(ax.get_ylim()) == (0.0, 1.0)
        assert ax.get_aspect() == 1.0

        display.set_style(
            ax_set_kwargs={
                "xlabel": "Custom X",
                "ylabel": "Custom Y",
                "xlim": (0.2, 0.8),
                "ylim": (0.1, 0.9),
            }
        )
        fig = display.plot()
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Custom X"
        assert ax.get_ylabel() == "Custom Y"
        assert tuple(ax.get_xlim()) == (0.2, 0.8)
        assert tuple(ax.get_ylim()) == (0.1, 0.9)
