import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from skore import CrossValidationReport, EstimatorReport


def _coefficient_features(report):
    names = report.inspection.coefficients().frame()["feature"].tolist()
    return [name for name in names if name != "Intercept"]


def _dataset_columns(report):
    if isinstance(report, CrossValidationReport):
        display = report.data.summarize(with_plots=False)
    else:
        display = report.data.summarize(data_source="test", with_plots=False)
    return list(display.summary["dataframe"].columns)


def test_numpy_feature_labels_agree():
    """NumPy coefficients and the data table use the same fallback labels."""
    X, y = make_regression(n_samples=40, n_features=3, random_state=0)
    report = EstimatorReport(
        LinearRegression(), X_train=X, y_train=y, X_test=X, y_test=y
    )
    expected = ["Feature #0", "Feature #1", "Feature #2"]
    assert _coefficient_features(report) == expected
    assert _dataset_columns(report)[:3] == expected


def test_integer_columns_are_stringified_on_a_copy():
    """Non-string dataframe columns are copied to strings before fit."""
    X, y = make_regression(n_samples=40, n_features=3, random_state=0)
    frame = pd.DataFrame(X)
    original_columns = list(frame.columns)

    report = EstimatorReport(
        LinearRegression(), X_train=frame, y_train=y, X_test=frame, y_test=y
    )
    assert list(frame.columns) == original_columns
    assert _coefficient_features(report) == ["0", "1", "2"]
    assert _dataset_columns(report)[:3] == ["0", "1", "2"]

    cv_report = CrossValidationReport(LinearRegression(), frame, y, splitter=2)
    assert list(frame.columns) == original_columns
    assert set(_coefficient_features(cv_report)) == {"0", "1", "2"}
    assert _dataset_columns(cv_report)[:3] == ["0", "1", "2"]


@pytest.mark.parametrize("container", ["pandas", "polars"])
def test_named_columns_stay_unchanged(container):
    """String column names are stored and displayed unchanged."""
    X, y = make_regression(n_samples=40, n_features=3, random_state=0)
    names = ["f0", "f1", "f2"]
    if container == "pandas":
        features = pd.DataFrame(X, columns=names)
        target = y
    else:
        features = pl.DataFrame({name: X[:, i] for i, name in enumerate(names)})
        target = pl.Series(y)

    report = EstimatorReport(
        LinearRegression(),
        X_train=features,
        y_train=target,
        X_test=features,
        y_test=target,
    )
    assert _coefficient_features(report) == names
    assert _dataset_columns(report)[:3] == names
