import pytest


def test_zero_frame(comparison_estimator_reports_binary_classification):
    """If `select_k` is zero then the frame is empty."""
    report = comparison_estimator_reports_binary_classification
    frame = report.inspection.impurity_decrease().frame(select_k=0)
    assert frame.empty


def test_zero_plot_raises(comparison_estimator_reports_binary_classification):
    """`plot(select_k=0)` raises ValueError."""
    report = comparison_estimator_reports_binary_classification
    display = report.inspection.impurity_decrease()
    with pytest.raises(ValueError, match="select_k=0 would produce an empty plot"):
        display.plot(select_k=0)


def test_negative(comparison_estimator_reports_binary_classification):
    """If `select_k` is negative then the features are the bottom `-select_k`."""
    report = comparison_estimator_reports_binary_classification
    frame = report.inspection.impurity_decrease().frame(select_k=-2)
    for _, group in frame.groupby("estimator", sort=False):
        assert group["feature"].nunique() == 2


def test_plot_with_select_k(pyplot, comparison_estimator_reports_binary_classification):
    """`select_k` works for plotting."""
    report = comparison_estimator_reports_binary_classification
    display = report.inspection.impurity_decrease()
    frame = display.frame(select_k=2)
    for _, group in frame.groupby("estimator", sort=False):
        assert group["feature"].nunique() == 2
    display.plot(select_k=2)
