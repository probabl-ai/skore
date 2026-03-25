import pytest

from skore import PermutationImportanceDisplay


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
class TestPermutationImportanceDisplay:
    def test_class_attributes(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.permutation_importance(n_repeats=2, seed=0)
        assert isinstance(display, PermutationImportanceDisplay)
        assert hasattr(display, "importances")
        assert hasattr(display, "report_type")

        fig = display.plot()
        assert fig is not None
        assert len(fig.axes) >= 1

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
        if "comparison" in fixture_prefix:
            expected.add("estimator")
        assert set(frame.columns) == expected

    def test_plot_structure(self, pyplot, fixture_prefix, task, request):
        _, axes = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        ax = axes[0]
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
        if "comparison" not in fixture_prefix:
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

        display.set_style(stripplot_kwargs={"height": 8})
        fig = display.plot()
        assert fig.get_figheight() == 8

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

    def test_frame_aggregate_none_ignores_level(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.permutation_importance(n_repeats=2, seed=0)
        for level in ["splits", "repetitions"]:
            frame = display.frame(aggregate=None, level=level)
            assert "repetition" in frame.columns
            assert "value" in frame.columns
            if "cross_validation" in fixture_prefix:
                assert "split" in frame.columns

    def test_unavailable_metric(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.permutation_importance(n_repeats=2, seed=0)

        err_msg = "The metric 'invalid-metric' is not available."
        with pytest.raises(ValueError, match=err_msg):
            display.frame(metric="invalid-metric")

    def test_frame_select_k(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.permutation_importance(n_repeats=2, seed=0)
        full = display.frame(sorting_order=None)
        sub = display.frame(select_k=2)
        group_cols = [
            c
            for c in ("estimator", "label", "output")
            if c in sub.columns and sub[c].nunique() > 1
        ]
        assert set(sub.columns) == set(full.columns)
        if group_cols:
            for _, group in sub.groupby(group_cols, sort=False, dropna=False):
                assert len(group) == 2
        else:
            assert len(sub) == 2
        assert set(sub["feature"]).issubset(set(full["feature"]))

    @pytest.mark.parametrize(
        "sorting_order",
        ["descending", "ascending"],
    )
    def test_frame_sorting_order(self, fixture_prefix, task, sorting_order, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.permutation_importance(n_repeats=2, seed=0)
        frame = display.frame(sorting_order=sorting_order)
        value_col = "value_mean" if "value_mean" in frame.columns else "value"
        group_cols = [
            c
            for c in frame.columns
            if c not in ("feature", value_col, "value_std", "data_source", "metric")
        ]
        if group_cols:
            groups = frame.groupby(group_cols, sort=False, dropna=False)
        else:
            groups = [(None, frame)]
        for _, group in groups:
            feature_order = group["feature"].unique()
            values = [
                group.loc[group["feature"] == f, value_col].mean()
                for f in feature_order
            ]
            expected = sorted(values, reverse=(sorting_order == "descending"))
            assert values == expected
