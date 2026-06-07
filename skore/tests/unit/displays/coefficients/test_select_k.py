import pytest


def test_zero_frame(comparison_estimator_reports_binary_classification):
    """If `select_k` is zero then the frame is empty."""
    report = comparison_estimator_reports_binary_classification
    frame = report.inspection.coefficients().frame(select_k=0)
    assert frame.empty


def test_zero_plot_raises(comparison_estimator_reports_binary_classification):
    """`plot(select_k=0)` raises ValueError."""
    report = comparison_estimator_reports_binary_classification
    display = report.inspection.coefficients()
    with pytest.raises(ValueError, match="select_k=0 would produce an empty plot"):
        display.plot(select_k=0)


def test_negative(comparison_estimator_reports_binary_classification):
    """If `select_k` is negative then the features are the bottom `-select_k`."""
    report = comparison_estimator_reports_binary_classification
    frame = report.inspection.coefficients().frame(select_k=-3)
    assert set(frame["feature"]) == {"Feature #3", "Feature #0", "Intercept"}


def test_different_features(
    pyplot, comparison_estimator_reports_binary_classification_different_features
):
    """`select_k` works for plotting when the estimators have different features."""
    report = comparison_estimator_reports_binary_classification_different_features

    display = report.inspection.coefficients()
    frame = display.frame(select_k=3)

    assert frame.query("estimator == 'LogisticRegression_1'")["feature"].tolist() == [
        "Feature #1",
        "Feature #2",
        "Feature #3",
    ]

    assert frame.query("estimator == 'LogisticRegression_2'")["feature"].tolist() == [
        "Intercept",
        "Feature #0",
        "Feature #1",
    ]
