import matplotlib as mpl
import pytest

from skore import ImpurityDecreaseDisplay


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports",
        "cross_validation_reports",
        "comparison_estimator_reports",
        "comparison_cross_validation_reports",
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
        fig = display.plot()
        assert fig is not None
        assert len(fig.axes) >= 1

    def test_frame_structure(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.impurity_decrease()
        frame = display.frame()

        expected = {"feature"}
        if "cross_validation" in fixture_prefix:
            expected |= {"importance_mean", "importance_std"}
        else:
            expected.add("importance")
        if "comparison" in fixture_prefix:
            expected.add("estimator")
        assert set(frame.columns) == expected

    def test_frame_aggregate(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.impurity_decrease()

        if "cross_validation" not in fixture_prefix:
            # situation 1: aggregate is ignored for non-CV report types
            assert set(display.frame().columns) == set(
                display.frame(aggregate=None).columns
            )
        else:
            # situation 2: aggregate=None does not aggregate
            frame_none = display.frame(aggregate=None)
            assert "split" in frame_none.columns
            assert "importance" in frame_none.columns
            assert "importance_mean" not in frame_none.columns
            assert "importance_std" not in frame_none.columns

            # situation 3: default aggregate aggregates properly
            frame_agg = display.frame()
            assert "split" not in frame_agg.columns
            assert "importance_mean" in frame_agg.columns
            assert "importance_std" in frame_agg.columns
            assert "importance" not in frame_agg.columns

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
        _, axes = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        ax = axes[0]
        assert ax.get_xlabel() == "Mean decrease in impurity"
        assert ax.get_ylabel() == ""

    def test_title(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.impurity_decrease()
        figure, _ = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        title = figure.get_suptitle()
        assert "Mean decrease in impurity (MDI)" in title
        if "comparison" not in fixture_prefix:
            estimator_name = display.importances["estimator"].iloc[0]
            assert estimator_name in title

    def test_kwargs(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.impurity_decrease()
        figure, _ = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        assert figure.get_figheight() == 6
        if "estimator" in fixture_prefix:
            display.set_style(barplot_kwargs={"height": 8})
            fig = display.plot()
        else:  # "cross_validation"
            display.set_style(stripplot_kwargs={"height": 8})
            fig = display.plot()
        assert fig.get_figheight() == 8

    def test_frame_select_k(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.impurity_decrease()
        full = display.frame(sorting_order=None)
        sub = display.frame(select_k=2)
        group_cols = [
            c for c in ("estimator",) if c in sub.columns and sub[c].nunique() > 1
        ]
        assert set(sub.columns) == set(full.columns)
        if group_cols:
            for _, group in sub.groupby(group_cols, sort=False):
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
        display = report.inspection.impurity_decrease()
        frame = display.frame(sorting_order=sorting_order)
        value_col = (
            "importance_mean" if "importance_mean" in frame.columns else "importance"
        )
        group_cols = [
            c
            for c in frame.columns
            if c not in ("feature", value_col, "importance_std")
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
    if "cross_validation" in fixture_prefix:
        assert set(frame.columns) >= {"feature", "importance_mean", "importance_std"}
    else:
        assert set(frame.columns) >= {"feature", "importance"}

    _, axes = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
    assert isinstance(axes[0], mpl.axes.Axes)
    assert axes[0].get_xlabel() == "Mean decrease in impurity"
