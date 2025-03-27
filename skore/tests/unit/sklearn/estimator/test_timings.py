import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skore import EstimatorReport


@pytest.fixture
def binary_classification_data():
    """Create a binary classification dataset and return fitted estimator and data."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return RandomForestClassifier(), {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def test_fit_time(binary_classification_data):
    """If the wrapped estimator is fitted inside of the EstimatorReport,
    then the fit time is a float."""
    estimator, data = binary_classification_data
    report = EstimatorReport(estimator, **data)

    assert isinstance(report.metrics.fit_time(), float)


def test_fit_time_estimator_already_fitted(binary_classification_data):
    """If the wrapped estimator was fitted outside of the EstimatorReport,
    then the fit time is None."""
    estimator, data = binary_classification_data
    estimator.fit(data["X_train"], data["y_train"])
    report = EstimatorReport(estimator, X_test=data["X_test"], y_test=data["y_test"])

    assert report.metrics.fit_time() is None


def test_fit_time_estimator_unfitted(binary_classification_data):
    """If the wrapped estimator is unfitted and fit=False, then the fit time is None."""
    estimator, data = binary_classification_data
    report = EstimatorReport(estimator, fit=False, **data)

    assert report.metrics.fit_time() is None


@pytest.mark.parametrize("data_source", ["test", "train", "X_y"])
def test_predict_time(data_source, binary_classification_data):
    estimator, data = binary_classification_data
    report = EstimatorReport(estimator, **data)

    X_, y_ = (data["X_test"], data["y_test"]) if data_source == "X_y" else (None, None)

    # Compute predictions
    report.metrics.accuracy(data_source=data_source, X=X_, y=y_)

    assert isinstance(
        report.metrics.predict_time(data_source=data_source, X=X_, y=y_), float
    )
