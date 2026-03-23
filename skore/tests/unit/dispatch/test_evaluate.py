import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

from skore import (
    ComparisonReport,
    CrossValidationReport,
    EstimatorReport,
    configuration,
    evaluate,
)


def test_evaluate_prefit_estimator(regression_data):
    """A fitted estimator with splitter='prefit' returns an EstimatorReport."""
    X, y = regression_data
    fitted = LinearRegression().fit(X, y)
    report = evaluate(fitted, X, y, splitter="prefit")
    assert isinstance(report, EstimatorReport)
    assert report.X_train is None
    assert report.y_train is None
    assert report.X_test is X
    assert report.y_test is y


def test_evaluate_prefit_unfitted_raises(regression_data):
    """An unfitted estimator with splitter='prefit' raises an error."""
    X, y = regression_data
    with pytest.raises(ValueError):
        evaluate(LinearRegression(), X, y, splitter="prefit")


def test_evaluate_float_splitter(regression_data):
    """A float splitter triggers a single split and returns an EstimatorReport."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=0.3)
    assert isinstance(report, EstimatorReport)
    assert len(report.X_test) == 0.3 * len(X)


def test_evaluate_default_splitter(regression_data):
    """Calling without splitter uses the default and returns an EstimatorReport."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    assert isinstance(report, EstimatorReport)
    assert len(report.X_test) == 0.2 * len(X)


def test_evaluate_int_splitter(regression_data):
    """An int splitter triggers cross-validation."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)
    assert isinstance(report, CrossValidationReport)
    assert len(report.estimator_reports_) == 3


def test_evaluate_cv_object_splitter(regression_data):
    """A CV object splitter triggers cross-validation."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=KFold(n_splits=4))
    assert isinstance(report, CrossValidationReport)
    assert len(report.estimator_reports_) == 4


def test_evaluate_multiple_estimators(regression_data):
    """A list of estimators returns a ComparisonReport."""
    X, y = regression_data
    estimators = [LinearRegression(), LinearRegression()]
    report = evaluate(estimators, X, y, splitter=0.2)
    assert isinstance(report, ComparisonReport)


def test_evaluate_multiple_estimators_prefit(regression_data):
    """A list of fitted estimators with splitter='prefit' returns ComparisonReport."""
    X, y = regression_data
    fitted1 = LinearRegression().fit(X, y)
    fitted2 = LinearRegression().fit(X, y)
    report = evaluate([fitted1, fitted2], X, y, splitter="prefit")
    assert isinstance(report, ComparisonReport)


def test_evaluate_multiple_estimators_cv(regression_data):
    """A list of estimators with int splitter returns a ComparisonReport."""
    X, y = regression_data
    estimators = [LinearRegression(), LinearRegression()]
    report = evaluate(estimators, X, y, splitter=3)
    assert isinstance(report, ComparisonReport)


def test_evaluate_multiple_estimators_multiple_X(regression_data):
    """A list of estimators with a list of X returns a ComparisonReport."""
    X, y = regression_data
    report = evaluate([LinearRegression(), LinearRegression()], [X, X], y, splitter=0.2)
    assert isinstance(report, ComparisonReport)


def test_evaluate_invalid_splitter_string(regression_data):
    """An invalid string splitter raises ValueError."""
    X, y = regression_data
    with pytest.raises(ValueError, match="Invalid string value for splitter"):
        evaluate(LinearRegression(), X, y, splitter="invalid")


def test_evaluate_classification(binary_classification_data):
    """evaluate works with a classification estimator."""
    X, y = binary_classification_data
    report = evaluate(LogisticRegression(), X, y, splitter=0.2)
    assert isinstance(report, EstimatorReport)


def test_evaluate_classification_cv(binary_classification_data):
    """evaluate works with classification + CV splitter object."""
    X, y = binary_classification_data
    report = evaluate(LogisticRegression(), X, y, splitter=StratifiedKFold(n_splits=3))
    assert isinstance(report, CrossValidationReport)
    assert len(report.estimator_reports_) == 3


def test_evaluate_pos_label(binary_classification_data):
    """pos_label is forwarded to the underlying report."""
    X, y = binary_classification_data
    report = evaluate(LogisticRegression(), X, y, splitter=0.2, pos_label=0)
    assert isinstance(report, EstimatorReport)
    assert report.pos_label == 0


def test_evaluate_follows_global_config_default(binary_classification_data):
    """Global diagnose config triggers diagnostics in evaluate."""
    X, y = binary_classification_data
    with configuration(diagnose=True):
        report = evaluate(LogisticRegression(), X, y, splitter=0.2)
    _results, checked_codes = report._diagnostics_cache
    assert len(checked_codes) > 0
