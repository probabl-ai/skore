import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skore import ComparisonReport, EstimatorReport


@pytest.fixture(scope="module")
def large_binary_classification_comparison_report():
    """Comparison of two estimator reports on a noisy binary classification
    dataset large enough to produce more than 500 distinct thresholds per
    child report (so the internal 500-point cap actually fires)."""
    X, y = make_classification(
        n_samples=4_000, n_features=4, flip_y=0.4, class_sep=0.3, random_state=0
    )
    report_1 = EstimatorReport(LogisticRegression(C=1.0).fit(X, y), X_test=X, y_test=y)
    report_2 = EstimatorReport(LogisticRegression(C=0.1).fit(X, y), X_test=X, y_test=y)
    return ComparisonReport([report_1, report_2])


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_curve_capped_to_500_points(
    large_binary_classification_comparison_report, metric
):
    """The ROC and precision-recall curve displays are downsampled to at most
    500 points per class and per child report on the public API. The fixture
    is calibrated so that the underlying scikit-learn curve has more than 500
    thresholds, so the assertion would fail if the cap was not enforced."""
    display = getattr(large_binary_classification_comparison_report.metrics, metric)()
    frame = display.frame()
    sizes = frame.groupby(["estimator", "label"], observed=True).size()
    assert sizes.max() == 500


def test_confusion_matrix_capped_to_500_thresholds(
    large_binary_classification_comparison_report,
):
    """The thresholded confusion matrix display is downsampled to at most
    500 thresholds per class and per child report on the public API. The
    fixture is calibrated so that the underlying scikit-learn curve has more
    than 500 thresholds, so the assertion would fail if the cap was not
    enforced."""
    display = large_binary_classification_comparison_report.metrics.confusion_matrix()
    df = display.confusion_matrix_thresholded
    assert df is not None
    n_thresholds = df.groupby(["estimator", "label"], observed=True)[
        "threshold"
    ].nunique()
    assert n_thresholds.max() == 500
