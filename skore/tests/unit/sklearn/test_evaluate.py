import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

from skore import ComparisonReport, CrossValidationReport, EstimatorReport, evaluate


@pytest.fixture
def regression_Xy():
    return make_regression(n_samples=80, n_features=5, random_state=42)


@pytest.fixture
def classification_Xy():
    return make_classification(n_samples=80, n_features=5, random_state=42)


# --- splitter="prefit" ---


def test_evaluate_prefit_estimator(regression_Xy):
    """A fitted estimator with splitter='prefit' returns an EstimatorReport."""
    X, y = regression_Xy
    fitted = LinearRegression().fit(X, y)
    report = evaluate(fitted, X, y, splitter="prefit")
    assert isinstance(report, EstimatorReport)


def test_evaluate_prefit_unfitted_raises(regression_Xy):
    """An unfitted estimator with splitter='prefit' raises an error."""
    X, y = regression_Xy
    with pytest.raises(ValueError):
        evaluate(LinearRegression(), X, y, splitter="prefit")


# --- splitter=float (train-test split) ---


def test_evaluate_float_splitter(regression_Xy):
    """A float splitter triggers a single split and returns an EstimatorReport."""
    X, y = regression_Xy
    report = evaluate(LinearRegression(), X, y, splitter=0.3)
    assert isinstance(report, EstimatorReport)
    assert report.X_train is not None
    assert report.X_test is not None


def test_evaluate_default_splitter(regression_Xy):
    """Calling without splitter uses the default and returns an EstimatorReport."""
    X, y = regression_Xy
    report = evaluate(LinearRegression(), X, y)
    assert isinstance(report, EstimatorReport)
    assert report.X_train is not None


# --- splitter=int (cross-validation) ---


def test_evaluate_int_splitter(regression_Xy):
    """An int splitter triggers cross-validation."""
    X, y = regression_Xy
    report = evaluate(LinearRegression(), X, y, splitter=3)
    assert isinstance(report, CrossValidationReport)
    assert len(report.estimator_reports_) == 3


# --- splitter=CV object ---


def test_evaluate_cv_object_splitter(regression_Xy):
    """A CV object splitter triggers cross-validation."""
    X, y = regression_Xy
    report = evaluate(LinearRegression(), X, y, splitter=KFold(n_splits=4))
    assert isinstance(report, CrossValidationReport)
    assert len(report.estimator_reports_) == 4


# --- multiple estimators ---


def test_evaluate_multiple_estimators(regression_Xy):
    """A list of estimators returns a ComparisonReport."""
    X, y = regression_Xy
    estimators = [LinearRegression(), LinearRegression()]
    report = evaluate(estimators, X, y, splitter=0.2)
    assert isinstance(report, ComparisonReport)


def test_evaluate_multiple_estimators_prefit(regression_Xy):
    """A list of fitted estimators with splitter='prefit' returns ComparisonReport."""
    X, y = regression_Xy
    fitted1 = LinearRegression().fit(X, y)
    fitted2 = LinearRegression().fit(X, y)
    report = evaluate([fitted1, fitted2], X, y, splitter="prefit")
    assert isinstance(report, ComparisonReport)


def test_evaluate_multiple_estimators_cv(regression_Xy):
    """A list of estimators with int splitter returns a ComparisonReport."""
    X, y = regression_Xy
    estimators = [LinearRegression(), LinearRegression()]
    report = evaluate(estimators, X, y, splitter=3)
    assert isinstance(report, ComparisonReport)


def test_evaluate_multiple_estimators_multiple_X(regression_Xy):
    """A list of estimators with a list of X returns a ComparisonReport."""
    X, y = regression_Xy
    report = evaluate([LinearRegression(), LinearRegression()], [X, X], y, splitter=0.2)
    assert isinstance(report, ComparisonReport)


# --- invalid splitter ---


def test_evaluate_invalid_splitter_string(regression_Xy):
    """An invalid string splitter raises ValueError."""
    X, y = regression_Xy
    with pytest.raises(ValueError, match="Invalid string value for splitter"):
        evaluate(LinearRegression(), X, y, splitter="invalid")


# --- classification ---


def test_evaluate_classification(classification_Xy):
    """evaluate works with a classification estimator."""
    X, y = classification_Xy
    report = evaluate(LogisticRegression(), X, y, splitter=0.2)
    assert isinstance(report, EstimatorReport)


def test_evaluate_classification_cv(classification_Xy):
    """evaluate works with classification + CV splitter object."""
    X, y = classification_Xy
    report = evaluate(LogisticRegression(), X, y, splitter=StratifiedKFold(n_splits=3))
    assert isinstance(report, CrossValidationReport)
    assert len(report.estimator_reports_) == 3


# --- pos_label ---


def test_evaluate_pos_label(classification_Xy):
    """pos_label is forwarded to the underlying report."""
    X, y = classification_Xy
    report = evaluate(LogisticRegression(), X, y, splitter=0.2, pos_label=1)
    assert isinstance(report, EstimatorReport)
