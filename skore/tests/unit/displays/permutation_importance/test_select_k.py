import pytest


def test_zero_frame(comparison_estimator_reports_binary_classification):
    """If `select_k` is zero then the frame is empty."""
    report = comparison_estimator_reports_binary_classification
    frame = report.inspection.permutation_importance(n_repeats=2, seed=0).frame(
        select_k=0
    )
    assert frame.empty


def test_zero_plot_raises(comparison_estimator_reports_binary_classification):
    """`plot(select_k=0)` raises ValueError."""
    report = comparison_estimator_reports_binary_classification
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    with pytest.raises(ValueError, match="select_k=0 would produce an empty plot"):
        display.plot(select_k=0)


def test_negative(comparison_estimator_reports_binary_classification):
    """If `select_k` is negative then the features are the bottom `-select_k`."""
    report = comparison_estimator_reports_binary_classification
    frame = report.inspection.permutation_importance(n_repeats=2, seed=0).frame(
        select_k=-3
    )
    for _, group in frame.groupby("estimator", sort=False):
        assert group["feature"].nunique() == 3


def test_different_features(
    pyplot,
    comparison_estimator_reports_binary_classification_different_features,
):
    """`select_k` works for plotting when the estimators have different features."""
    report = comparison_estimator_reports_binary_classification_different_features
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    frame = display.frame(select_k=3)

    for _, group in frame.groupby("estimator", sort=False):
        assert group["feature"].nunique() <= 3

    display.plot(select_k=3)
