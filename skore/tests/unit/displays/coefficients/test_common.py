import numpy as np
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

        fig = display.plot()
        assert fig is not None
        assert len(fig.axes) >= 1

    def test_frame_structure(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.coefficients()
        frame = display.frame()

        if "cross_validation" in fixture_prefix:
            expected = {"feature", "coefficient_mean", "coefficient_std"}
        else:
            expected = {"feature", "coefficient"}
        if "comparison" in fixture_prefix:
            expected.add("estimator")
        if task == "multiclass_classification":
            expected.add("label")
        if task == "multioutput_regression":
            expected.add("output")
        assert set(frame.columns) == expected

    def test_frame_aggregate(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.coefficients()

        if "cross_validation" not in fixture_prefix:
            # situation 1: aggregate is ignored for non-CV report types
            assert set(display.frame().columns) == set(
                display.frame(aggregate=None).columns
            )
        else:
            # situation 2: aggregate=None does not aggregate
            frame_none = display.frame(aggregate=None)
            assert "split" in frame_none.columns
            assert "coefficient" in frame_none.columns
            assert "coefficient_mean" not in frame_none.columns
            assert "coefficient_std" not in frame_none.columns

            # situation 3: default aggregate aggregates properly
            frame_agg = display.frame()
            assert "split" not in frame_agg.columns
            assert "coefficient_mean" in frame_agg.columns
            assert "coefficient_std" in frame_agg.columns
            assert "coefficient" not in frame_agg.columns

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
            "coefficient",
        }
        assert set(display.coefficients.columns) == expected

    def test_plot_structure(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        _, axes = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        ax = axes[0]
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""

    def test_title(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.coefficients()
        figure, _ = request.getfixturevalue(f"{fixture_prefix}_{task}_figure_axes")
        title = figure.get_suptitle()
        assert "Coefficient" in title
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
            display.set_style(barplot_kwargs={"height": 8})
            fig = display.plot()
        else:
            display.set_style(stripplot_kwargs={"height": 8})
            fig = display.plot()
        assert fig.get_figheight() == 8

    def test_frame_select_k(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.coefficients()
        full = display.frame(sorting_order=None)
        sub = display.frame(select_k=2)
        group_cols = [
            c for c in ("split", "estimator", "label", "output") if c in sub.columns
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
        display = report.inspection.coefficients()
        frame = display.frame(sorting_order=sorting_order)
        group_cols = [
            c
            for c in frame.columns
            if c
            not in (
                "feature",
                "coefficient",
                "coefficient_mean",
                "coefficient_std",
                "split",
            )
        ]
        if group_cols:
            groups = frame.groupby(group_cols, sort=False, observed=True)
        else:
            groups = [(None, frame)]
        for _, group in groups:
            if "cross_validation" in fixture_prefix:
                feature_order = group["feature"].unique()
                coefs = [
                    group.loc[group["feature"] == f, "coefficient_mean"].iloc[0]
                    for f in feature_order
                ]
            else:
                coefs = group["coefficient"].tolist()
            expected = sorted(coefs, key=abs, reverse=(sorting_order == "descending"))
            np.testing.assert_array_equal(coefs, expected)

    def test_include_intercept(self, fixture_prefix, task, request):
        """Check whether or not we can include or exclude the intercept."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}")
        if isinstance(report, tuple):
            report = report[0]
        display = report.inspection.coefficients()
        assert (
            display.frame(include_intercept=False).query("feature == 'Intercept'").empty
        )
