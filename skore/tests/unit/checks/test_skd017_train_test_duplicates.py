import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

from skore import EstimatorReport, TrainTestSplit, evaluate
from skore._checks.skd017_train_test_duplicates import CheckTrainTestDuplicates
from skore._checks.utils import CheckNotApplicable

report_types = pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])


def _make_report(report_type, estimator, X, y):
    splitter = (
        TrainTestSplit(random_state=0)
        if report_type == "estimator"
        else KFold(n_splits=3, shuffle=True, random_state=0)
    )
    return evaluate(estimator, X, y, splitter=splitter)


def _columns(X):
    return [f"feature_{i}" for i in range(X.shape[1])]


def _to_numpy(X):
    return X


def _to_pandas(X):
    return pd.DataFrame(X, columns=_columns(X))


def _to_polars(X):
    pl = pytest.importorskip("polars")
    return pl.DataFrame(X, schema=_columns(X))


@report_types
@pytest.mark.parametrize("to_native", [_to_numpy, _to_pandas, _to_polars])
def test_duplicates_detected(regression_data, report_type, to_native):
    """Duplicate rows in the dataset trigger SKD017."""
    X, y = (np.asarray(a) for a in regression_data)
    X = np.vstack([X, X[:5]])
    y = np.concatenate([y, y[:5]])
    report = _make_report(report_type, LinearRegression(), to_native(X), y)
    explanation = CheckTrainTestDuplicates().check_function(report)
    assert explanation is not None
    assert "10 row(s)" in explanation
    assert f"{100 * 10 / len(X):.1f}%" in explanation


@report_types
def test_no_duplicates_passes(regression_data, report_type):
    """A dataset with no duplicate rows does not trigger SKD017."""
    X, y = (np.asarray(a) for a in regression_data)
    report = _make_report(report_type, LinearRegression(), X, y)
    assert CheckTrainTestDuplicates().check_function(report) is None


@report_types
def test_duplicates_detected_multi_output(regression_data, report_type):
    """Duplicates are detected when the target has several columns."""
    X, y = (np.asarray(a) for a in regression_data)
    y = np.column_stack([y, y**2])
    X = np.vstack([X, X[:5]])
    y = np.vstack([y, y[:5]])
    report = _make_report(report_type, LinearRegression(), X, y)
    explanation = CheckTrainTestDuplicates().check_function(report)
    assert explanation is not None
    assert "10 row(s)" in explanation


def test_not_applicable_when_x_train_is_none(regression_data):
    """SKD017 raises CheckNotApplicable when X_train is unavailable."""
    X, y = (np.asarray(a) for a in regression_data)
    n_train = int(0.8 * len(X))
    estimator = LinearRegression().fit(X[:n_train], y[:n_train])
    report = EstimatorReport(estimator, X_test=X[n_train:], y_test=y[n_train:])
    with pytest.raises(CheckNotApplicable, match="Train data is unavailable"):
        CheckTrainTestDuplicates().check_function(report)


@report_types
def test_duplicates_with_nan_are_detected(regression_data, report_type):
    """Duplicate rows containing NaN still count as duplicates."""
    X, y = (np.asarray(a) for a in regression_data)
    X = _to_pandas(X)
    X.iloc[0, 0] = np.nan
    X = pd.concat([X, X.iloc[:1]], ignore_index=True)
    y = np.concatenate([y, y[:1]])
    report = _make_report(report_type, HistGradientBoostingRegressor(), X, y)
    explanation = CheckTrainTestDuplicates().check_function(report)
    assert explanation is not None
    assert "2 row(s)" in explanation


@report_types
def test_identical_features_different_target_not_counted(regression_data, report_type):
    """Rows sharing features but not target are not counted as duplicates."""
    X, y = (np.asarray(a) for a in regression_data)
    X = np.vstack([X, X[:1]])
    y = np.concatenate([y, y[:1] + 1])
    report = _make_report(report_type, LinearRegression(), X, y)
    assert CheckTrainTestDuplicates().check_function(report) is None


def test_not_applicable_when_rows_cannot_be_compared():
    """Uncomparable (list-valued) cells degrade to CheckNotApplicable."""
    # The transformer drops the list column so the estimator can fit, while the
    # report still stores the raw column that cannot be de-duplicated.
    pipe = make_pipeline(
        ColumnTransformer([("num", "passthrough", ["b"])], remainder="drop"),
        LogisticRegression(),
    )
    X_train = pd.DataFrame({"a": [[1, 2], [3, 4], [5, 6]], "b": [1.0, 2.0, 3.0]})
    X_test = pd.DataFrame({"a": [[1, 2]], "b": [1.0]})
    report = EstimatorReport(
        pipe,
        X_train=X_train,
        y_train=[0, 1, 0],
        X_test=X_test,
        y_test=[0],
    )
    with pytest.raises(CheckNotApplicable, match="Could not compare rows"):
        CheckTrainTestDuplicates().check_function(report)


@report_types
def test_target_name_collides_with_feature(regression_data, report_type):
    """A target named like a feature column does not break the duplicate check."""
    X, _ = (np.asarray(a) for a in regression_data)
    X = _to_pandas(X)
    y = pd.Series(X["feature_0"].to_numpy(), name="feature_0")
    report = _make_report(report_type, LinearRegression(), X, y)
    assert CheckTrainTestDuplicates().check_function(report) is None
