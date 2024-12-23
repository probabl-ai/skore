import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from skore import CrossValidationReport


@pytest.fixture
def binary_classification_data():
    """Create a binary classification dataset and return fitted estimator and data."""
    X, y = make_classification(random_state=42)
    return RandomForestClassifier(), X, y


@pytest.fixture
def multiclass_classification_data():
    """Create a multiclass classification dataset and return fitted estimator and
    data."""
    X, y = make_classification(
        n_classes=3, n_clusters_per_class=1, random_state=42, n_informative=10
    )
    return RandomForestClassifier(), X, y


@pytest.fixture
def regression_data():
    """Create a regression dataset and return fitted estimator and data."""
    X, y = make_regression(random_state=42)
    return LinearRegression(), X, y


@pytest.fixture
def regression_multioutput_data():
    """Create a regression dataset and return fitted estimator and data."""
    X, y = make_regression(n_targets=2, random_state=42)
    return LinearRegression(), X, y


@pytest.mark.parametrize(
    "metric", ["accuracy", "precision", "recall", "roc_auc", "brier_score", "log_loss"]
)
def test_cross_validation_report_binary(binary_classification_data, metric):
    estimator, X, y = binary_classification_data
    reporter = CrossValidationReport(estimator, X, y, cv=5)
    assert hasattr(reporter.metrics, metric)
    result = getattr(reporter.metrics, metric)()
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 1)


@pytest.mark.parametrize(
    "metric", ["accuracy", "precision", "recall", "roc_auc", "log_loss"]
)
def test_cross_validation_report_multiclass(multiclass_classification_data, metric):
    estimator, X, y = multiclass_classification_data
    reporter = CrossValidationReport(estimator, X, y, cv=5)
    assert hasattr(reporter.metrics, metric)
    result = getattr(reporter.metrics, metric)()
    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize("metric", ["r2", "rmse"])
def test_cross_validation_report_regression(regression_data, metric):
    estimator, X, y = regression_data
    reporter = CrossValidationReport(estimator, X, y, cv=5)
    assert hasattr(reporter.metrics, metric)
    result = getattr(reporter.metrics, metric)()
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 1)


@pytest.mark.parametrize("metric", ["r2", "rmse"])
@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average"])
def test_cross_validation_report_multioutput_regression(
    regression_multioutput_data, metric, multioutput
):
    estimator, X, y = regression_multioutput_data
    reporter = CrossValidationReport(estimator, X, y, cv=5)
    assert hasattr(reporter.metrics, metric)
    result = getattr(reporter.metrics, metric)(multioutput=multioutput)
    assert isinstance(result, pd.DataFrame)
    if multioutput == "raw_values":
        assert result.shape == (5, y.shape[1])
    else:
        assert result.shape == (5, 1)
