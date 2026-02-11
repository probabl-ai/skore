import pytest


@pytest.mark.parametrize("sorting_order", ["descending", "ascending"])
def test_frame_sorts_per_estimator(
    comparison_estimator_reports_binary_classification, sorting_order
):
    """`sorting_order` sorts per estimator."""
    report = comparison_estimator_reports_binary_classification
    frame = report.inspection.coefficients().frame(sorting_order=sorting_order)

    for estimator in set(frame["estimator"]):
        coefs = list(frame.query(f"estimator == '{estimator}'")["coefficient"])
        assert coefs == sorted(coefs, key=abs, reverse=(sorting_order == "descending"))


@pytest.mark.parametrize(
    "sorting_order, expected_features",
    [
        (
            "descending",
            [
                "Feature #2",
                "Feature #1",
                "Feature #3",
                "Intercept",
                "Feature #0",
                "Feature #2",
                "Feature #1",
                "Feature #3",
                "Intercept",
                "Feature #0",
            ],
        ),
        (
            "ascending",
            [
                "Feature #0",
                "Intercept",
                "Feature #3",
                "Feature #1",
                "Feature #2",
                "Feature #0",
                "Intercept",
                "Feature #3",
                "Feature #1",
                "Feature #2",
            ],
        ),
    ],
)
def test_plot_different_features(
    pyplot,
    comparison_estimator_reports_binary_classification,
    sorting_order,
    expected_features,
):
    """`sorting_order` works for plotting when estimators have different features."""
    report = comparison_estimator_reports_binary_classification
    display = report.inspection.coefficients()
    frame = display.frame(sorting_order=sorting_order)

    assert frame["feature"].tolist() == expected_features
