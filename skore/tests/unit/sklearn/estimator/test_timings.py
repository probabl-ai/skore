import joblib
import pandas as pd
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


def test_only_fit(binary_classification_data):
    estimator, data = binary_classification_data
    report = EstimatorReport(estimator, **data)

    result = report.metrics.timings()
    assert isinstance(result, dict)
    assert len(result) == 1
    assert isinstance(result.get("fit_time"), float)


def test_only_fit_unfitted(binary_classification_data):
    """If the wrapped estimator is unfitted and fit=False,
    then the fit time is not included in timings."""
    estimator, data = binary_classification_data
    report = EstimatorReport(estimator, fit=False, **data)

    result = report.metrics.timings()
    assert result == {}


@pytest.mark.parametrize("data_source", ["test", "train", "X_y"])
def test_predict_prefitted(data_source, binary_classification_data):
    """If the wrapped estimator is prefitted, and some predictions are computed,
    then `timings` has one key per prediction data source."""
    estimator, data = binary_classification_data
    report = EstimatorReport(estimator.fit(data["X_train"], data["y_train"]), **data)

    if data_source == "X_y":
        X_, y_ = (data["X_test"], data["y_test"])
        data_source_hash = joblib.hash((data["X_test"], data["y_test"]))
        data_source_check = f"X_y_{data_source_hash}"
    else:
        X_, y_ = (None, None)
        data_source_check = data_source

    # Compute predictions on data source
    report.metrics.accuracy(data_source=data_source, X=X_, y=y_)

    result = report.metrics.timings()
    assert isinstance(result, dict)
    assert len(result) == 1
    assert isinstance(result.get(f"predict_time_{data_source_check}"), float)


def test_everything(binary_classification_data):
    estimator, data = binary_classification_data
    report = EstimatorReport(estimator, **data)

    # Compute predictions on each data source
    report.metrics.accuracy(data_source="train", X=None, y=None)
    report.metrics.accuracy(data_source="test", X=None, y=None)
    report.metrics.accuracy(data_source="X_y", X=data["X_test"], y=data["y_test"])
    data_source_hash = joblib.hash((data["X_test"], data["y_test"]))
    data_source_check = f"X_y_{data_source_hash}"

    result = report.metrics.timings()
    assert isinstance(result, dict)
    assert len(result) == 4
    assert isinstance(result.get("fit_time"), float)
    assert isinstance(result.get("predict_time_train"), float)
    assert isinstance(result.get("predict_time_test"), float)
    assert isinstance(result.get(f"predict_time_{data_source_check}"), float)


def test_fit_time(binary_classification_data):
    """If the wrapped estimator is fitted inside of the EstimatorReport,
    then the fit time is a float."""
    estimator, data = binary_classification_data
    report = EstimatorReport(estimator, **data)

    assert isinstance(report.metrics._fit_time(), float)


def test_fit_time_estimator_already_fitted(binary_classification_data):
    """If the wrapped estimator was fitted outside of the EstimatorReport,
    then the fit time is None."""
    estimator, data = binary_classification_data
    estimator.fit(data["X_train"], data["y_train"])
    report = EstimatorReport(estimator, X_test=data["X_test"], y_test=data["y_test"])

    assert report.metrics._fit_time() is None


def test_fit_time_estimator_unfitted(binary_classification_data):
    """If the wrapped estimator is unfitted and fit=False, then the fit time is None."""
    estimator, data = binary_classification_data
    report = EstimatorReport(estimator, fit=False, **data)

    assert report.metrics._fit_time() is None


@pytest.mark.parametrize("data_source", ["test", "train", "X_y"])
def test_predict_time(data_source, binary_classification_data):
    estimator, data = binary_classification_data
    report = EstimatorReport(estimator, **data)

    X_, y_ = (data["X_test"], data["y_test"]) if data_source == "X_y" else (None, None)

    # Compute predictions
    report.metrics.accuracy(data_source=data_source, X=X_, y=y_)

    assert isinstance(
        report.metrics._predict_time(data_source=data_source, X=X_, y=y_), float
    )


def test_report_metrics_fit_time(binary_classification_data):
    estimator, data = binary_classification_data
    report = EstimatorReport(estimator, **data)

    assert isinstance(report.metrics.report_metrics(scoring=["fit_time"]), pd.DataFrame)


@pytest.mark.parametrize("data_source", ["test", "train", "X_y"])
def test_report_metrics_predict_time(data_source, binary_classification_data):
    estimator, data = binary_classification_data
    report = EstimatorReport(estimator, **data)

    X_, y_ = (data["X_test"], data["y_test"]) if data_source == "X_y" else (None, None)

    assert isinstance(
        report.metrics.report_metrics(
            scoring=["predict_time"], data_source=data_source, X=X_, y=y_
        ),
        pd.DataFrame,
    )
