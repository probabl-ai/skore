import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from skore import ComparisonReport, CrossValidationReport, EstimatorReport


@pytest.fixture
def comparison_report(binary_classification_train_test_split):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    report_1 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    return ComparisonReport(reports={"report_1": report_1, "report_2": report_2})


def test_estimator(logistic_binary_classification_with_train_test):
    """Test that select_k works with EstimatorReport."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    coefficients = report.feature_importance.coefficients().frame(select_k=3)

    assert set(coefficients["feature"]) == {"Feature #10", "Feature #1", "Feature #15"}


def test_comparison_cross_validation(logistic_binary_classification_data):
    """Test that select_k works with ComparisonReports of CrossValidationReports."""
    estimator, X, y = logistic_binary_classification_data
    report_1 = CrossValidationReport(estimator, X, y)
    report_2 = CrossValidationReport(estimator, X, y)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    coefficients = report.feature_importance.coefficients().frame(select_k=3)

    assert set(coefficients["feature"]) == {"Feature #10", "Feature #1", "Feature #15"}


def test_zero(comparison_report):
    """Test that if select_k is zero then the output is an empty dataframe."""

    frame = comparison_report.feature_importance.coefficients().frame(select_k=0)
    assert frame.empty


def test_negative(comparison_report):
    """Test that if select_k is negative then the features are the bottom k."""
    frame = comparison_report.feature_importance.coefficients().frame(select_k=-3)
    assert set(frame["feature"]) == {"Feature #17", "Feature #16", "Feature #0"}


def test_multiclass(multiclass_classification_train_test_split):
    """Test that select_k works per estimator and per class in multiclass comparison."""
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    report_1 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    frame = report.feature_importance.coefficients().frame(select_k=2)

    assert {
        (report, int(label)): list(group["feature"])
        for (report, label), group in frame.groupby(["estimator", "label"])
    } == {
        ("report_1", 0): ["Intercept", "Feature #7"],
        ("report_1", 1): ["Feature #6", "Feature #8"],
        ("report_1", 2): ["Intercept", "Feature #5"],
        ("report_2", 0): ["Intercept", "Feature #7"],
        ("report_2", 1): ["Feature #6", "Feature #8"],
        ("report_2", 2): ["Intercept", "Feature #5"],
    }


def test_plot(comparison_report):
    """Test that plot method correctly uses select_k parameter."""
    display = comparison_report.feature_importance.coefficients()

    display.plot(select_k=3)

    labels = [
        tick_label.get_text() for tick_label in display.ax_.get_yaxis().get_ticklabels()
    ]
    assert labels == ["Feature #10", "Feature #1", "Feature #15"]


def test_plot_different_features(logistic_binary_classification_with_train_test):
    """
    Test that select_k works correctly when the estimators have different features.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report_1 = EstimatorReport(
        estimator, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    report_2 = EstimatorReport(
        Pipeline([("poly", PolynomialFeatures()), ("predictor", clone(estimator))]),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    display = report.feature_importance.coefficients()
    display.plot(select_k=3)

    labels = [
        [tick_label.get_text() for tick_label in ax.get_yaxis().get_ticklabels()]
        for ax in display.ax_
    ]
    assert labels == [
        ["Feature #10", "Feature #1", "Feature #15"],
        ["Intercept", "x10", "x1"],
    ]
