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
def multi_classification_comparator():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
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
    display_summary = regression_comparator.metrics.summarize()
    display_summary.plot_comparison_estimator("r2", "fit_time")
    assert display_summary.ax_.get_xlabel() == "R²"
    assert display_summary.ax_.get_ylabel() == "Fit time (s) on train set"
    assert len(display_summary.ax_.get_title()) > 4


def test_error_invalid_metric(regression_comparator):
    comp = regression_comparator
    with pytest.raises(ValueError):
        comp.metrics.summarize().plot_comparison_estimator(
            "invalid_metric", "invalid_metric_bis"
        )
