import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from skore import ComparisonReport, EstimatorReport


@pytest.fixture
def binary_classification_comparator():
    X, y = make_classification(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    report_1 = EstimatorReport(
        estimator=HistGradientBoostingClassifier(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        estimator=LogisticRegression(max_iter=50),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    comp = ComparisonReport({"report_1": report_1, "report_2": report_2})
    return comp


@pytest.fixture
def regression_comparator():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    report_1 = EstimatorReport(
        estimator=HistGradientBoostingRegressor(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        estimator=LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    comp = ComparisonReport({"report_1": report_1, "report_2": report_2})
    return comp


def test_regression_comparator(regression_comparator):
    """Test that the regression comparator can summarize metrics and plot them."""
    display_summary = regression_comparator.metrics.summarize()
    display_summary.plot(x="r2", y="fit_time")
    assert display_summary.ax_.get_xlabel() == "R²"
    assert display_summary.ax_.get_ylabel() == "Fit time (s) on train set"
    assert len(display_summary.ax_.get_title()) > 4


def test_data_source_affect_title_and_axis(regression_comparator):
    """Test that the data source does change the title and axis labels."""
    comp = regression_comparator
    display_summary = comp.metrics.summarize(data_source="train")
    display_summary.plot(x="r2", y="fit_time")
    assert "on train set" in display_summary.ax_.get_title()
    assert "on test set" not in display_summary.ax_.get_ylabel()


def test_error_invalid_metric(regression_comparator):
    """Test the error raised when an invalid metric is used."""
    comp = regression_comparator
    with pytest.raises(ValueError):
        comp.metrics.summarize().plot(x="invalid_metric", y="fit_time")
    with pytest.raises(ValueError):
        comp.metrics.summarize().plot(x="fit_time", y="invalid_metric")


def test_needs_positive_label(binary_classification_comparator):
    """
    Test the error raised when a metric requiring a positive label is selected,
    without giving the pos_label.
    """
    comp = binary_classification_comparator
    with pytest.raises(
        ValueError,
        match="The perf metric x requires to add a positive label parameter.",
    ):
        comp.metrics.summarize().plot(x="precision", y="fit_time")
    with pytest.raises(
        ValueError,
        match="The perf metric y requires to add a positive label parameter.",
    ):
        comp.metrics.summarize().plot(x="fit_time", y="precision")


def test_no_positive_label_unrequired(binary_classification_comparator):
    """
    Test that no error is raised when a metric not requiring a positive label is
    selected.
    """
    display_summary = binary_classification_comparator.metrics.summarize()
    display_summary.plot(x="brier_score", y="fit_time")
    assert display_summary.ax_.get_xlabel() == "Brier score"
    assert display_summary.ax_.get_ylabel() == "Fit time (s) on train set"
    assert len(display_summary.ax_.get_title()) > 4

    display_summary = binary_classification_comparator.metrics.summarize()
    display_summary.plot(x="fit_time", y="brier_score")
    assert display_summary.ax_.get_xlabel() == "Fit time (s) on train set"
    assert display_summary.ax_.get_ylabel() == "Brier score"
    assert len(display_summary.ax_.get_title()) > 4
