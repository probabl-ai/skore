import matplotlib as mpl
import pandas as pd
import pytest

from skore import ImpurityDecreaseDisplay


def test_impurity_decrease_display_invalid_report_type():
    importances = pd.DataFrame(
        {
            "estimator": ["estimator1"],
            "feature": ["feature1"],
            "importance": [1.0],
        }
    )
    display = ImpurityDecreaseDisplay(
        importances=importances, report_type="invalid-type"
    )
    with pytest.raises(TypeError, match="Unexpected report type: 'invalid-type'"):
        display.frame()


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports",
        "cross_validation_reports",
    ],
)
@pytest.mark.parametrize("task", ["binary_classification", "regression"])
class TestImpurityDecreaseDisplay:
    def test_class_attributes(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.impurity_decrease()
        assert isinstance(display, ImpurityDecreaseDisplay)
        assert hasattr(display, "importances")
        assert hasattr(display, "report_type")
        display.plot()
        assert hasattr(display, "figure_")
        assert hasattr(display, "ax_")
        if fixture_prefix == "cross_validation_reports":
            assert hasattr(display, "facet_")

    def test_frame_structure(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.impurity_decrease()
        frame = display.frame()
        expected = {"feature", "importance"}
        if fixture_prefix == "cross_validation_reports":
            expected.add("split")
        assert set(frame.columns) == expected

    def test_internal_data_structure(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.impurity_decrease()
        expected = {"estimator", "split", "feature", "importance"}
        assert set(display.importances.columns) == expected

    def test_plot_structure(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        _, ax = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        if hasattr(ax, "flatten"):
            ax = ax.flatten()[0]
        assert ax.get_xlabel() == "Mean Decrease in Impurity (MDI)"
        assert ax.get_ylabel() == ""

    def test_title(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.impurity_decrease()
        figure, _ = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        title = figure.get_suptitle()
        assert "Mean Decrease in Impurity (MDI)" in title
        estimator_name = display.importances["estimator"].iloc[0]
        assert estimator_name in title

    def test_kwargs(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.impurity_decrease()
        figure, _ = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        assert figure.get_figheight() == 6
        if fixture_prefix == "estimator_reports":
            display.set_style(barplot_kwargs={"height": 8}).plot()
        else:
            display.set_style(stripplot_kwargs={"height": 8}).plot()
        assert display.figure_.get_figheight() == 8


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports",
        "cross_validation_reports",
    ],
)
@pytest.mark.parametrize(
    "task", ["multiclass_classification", "multioutput_regression"]
)
def test_multiclass_and_multioutput(pyplot, fixture_prefix, task, request):
    report = request.getfixturevalue(f"{fixture_prefix}_{task}")
    if isinstance(report, tuple):
        report = report[0]
    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)
    assert set(display.importances.columns) == {
        "estimator",
        "split",
        "feature",
        "importance",
    }
    frame = display.frame()
    assert set(frame.columns) >= {"feature", "importance"}
    if fixture_prefix == "cross_validation_reports":
        assert "split" in frame.columns

    _, ax = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
    assert isinstance(ax, mpl.axes.Axes)
    assert ax.get_xlabel() == "Mean Decrease in Impurity (MDI)"
