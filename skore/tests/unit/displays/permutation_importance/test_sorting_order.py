import pytest


@pytest.mark.parametrize("sorting_order", ["descending", "ascending"])
def test_frame_sorts_per_estimator(
    comparison_estimator_reports_binary_classification, sorting_order
):
    """`sorting_order` sorts per estimator."""
    report = comparison_estimator_reports_binary_classification
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    frame = display.frame(sorting_order=sorting_order)

    value_col = "value_mean" if "value_mean" in frame.columns else "value"
    for _, group in frame.groupby(
        [c for c in ["estimator", "label", "output"] if c in frame.columns],
        sort=False,
        dropna=False,
    ):
        feature_order = group["feature"].unique()
        values = [
            group.loc[group["feature"] == f, value_col].mean() for f in feature_order
        ]
        expected = sorted(values, reverse=(sorting_order == "descending"))
        assert values == expected


@pytest.mark.parametrize("sorting_order", ["descending", "ascending"])
def test_plot_with_sorting_order(
    pyplot, comparison_estimator_reports_binary_classification, sorting_order
):
    """`sorting_order` works for plotting."""
    report = comparison_estimator_reports_binary_classification
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    fig = display.plot(sorting_order=sorting_order)
    assert fig is not None
