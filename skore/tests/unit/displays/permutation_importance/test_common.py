import numpy as np
import pandas as pd
import pytest

from skore import PermutationImportanceDisplay


@pytest.mark.parametrize("method", ["frame", "plot"])
def test_invalid_report_type(pyplot, method):
    importances = pd.DataFrame(
        {
            "estimator": ["estimator"],
            "data_source": ["test"],
            "metric": ["r2"],
            "split": [np.nan],
            "feature": ["feature1"],
            "label": [np.nan],
            "output": [np.nan],
            "repetition": [0],
            "value": [1.0],
        }
    )
    display = PermutationImportanceDisplay(
        importances=importances, report_type="invalid-type"
    )
    with pytest.raises(TypeError, match="Unexpected report type: 'invalid-type'"):
        getattr(display, method)()


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports",
        "cross_validation_reports",
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
class TestPermutationImportanceDisplay:
    def test_class_attributes(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.permutation_importance(n_repeats=2, seed=0)
        assert isinstance(display, PermutationImportanceDisplay)
        assert hasattr(display, "importances")
        assert hasattr(display, "report_type")

        display.plot()
        assert hasattr(display, "facet_")
        assert hasattr(display, "figure_")
        assert hasattr(display, "ax_")

    def test_internal_data_structure(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.permutation_importance(n_repeats=2, seed=0)
        expected = [
            "estimator",
            "data_source",
            "metric",
            "split",
            "feature",
            "label",
            "output",
            "repetition",
            "value",
        ]
        assert display.importances.columns.tolist() == expected

    def test_frame_structure(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.permutation_importance(n_repeats=2, seed=0)
        frame = display.frame()

        expected = {"data_source", "metric", "feature", "value_mean", "value_std"}
        if "cross_validation" in fixture_prefix:
            expected.add("split")
        assert set(frame.columns) == expected

    def test_plot_structure(self, pyplot, fixture_prefix, task, request):
        _, ax = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        if hasattr(ax, "flatten"):
            ax = ax.flatten()[0]
        assert "Decrease in" in ax.get_xlabel()
        assert ax.get_ylabel() == ""

    def test_title(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.permutation_importance(n_repeats=2, seed=0)
        figure, _ = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        title = figure.get_suptitle()
        assert "Permutation importance" in title
        estimator_name = display.importances["estimator"].iloc[0]
        assert estimator_name in title
        if "cross_validation" in fixture_prefix:
            assert "averaged over splits" in title
        else:
            assert "averaged over splits" not in title

    def test_set_style(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.permutation_importance(n_repeats=2, seed=0)
        figure, _ = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        assert figure.get_figheight() == 6

        display.set_style(stripplot_kwargs={"height": 8}).plot()
        assert display.figure_.get_figheight() == 8

    @pytest.mark.parametrize(
        "aggregate, expected_value_columns",
        [
            ("mean", ["value"]),
            ("std", ["value"]),
            (("mean", "std"), ["value_mean", "value_std"]),
        ],
    )
    def test_frame_aggregate(
        self, fixture_prefix, task, aggregate, expected_value_columns, request
    ):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.permutation_importance(n_repeats=2, seed=0)
        frame = display.frame(aggregate=aggregate)
        for col in expected_value_columns:
            assert col in frame.columns
