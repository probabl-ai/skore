import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from skrub import DatetimeEncoder

from skore import EstimatorReport, evaluate
from skore._checks.skd013_train_test_time_overlap import CheckTrainTestTimeOverlap
from skore._checks.utils import CheckNotApplicable


def _datetime_pipeline():
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("date", DatetimeEncoder(), ["date"])],
                    remainder="passthrough",
                ),
            ),
            ("reg", LinearRegression()),
        ]
    )


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_train_test_time_overlap(report_type):
    """Shuffled split/CV triggers overlap."""
    n = 200
    X = pd.DataFrame(
        {
            "feat": np.arange(n, dtype=float),
            "date": pd.date_range("2026-12-01", periods=n, freq="D"),
        }
    )
    y = np.arange(n, dtype=float)
    pipe = _datetime_pipeline()

    if report_type == "estimator":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=0
        )
        report = EstimatorReport(
            pipe, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
    else:
        report = evaluate(
            pipe, X, y, splitter=KFold(n_splits=5, shuffle=True, random_state=0)
        )
    explanation = CheckTrainTestTimeOverlap().check_function(report)
    assert explanation is not None
    assert "date" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_train_test_time_no_overlap(report_type):
    """Proper temporal train/test split or time-series CV doesn't trigger SKD013."""
    n = 200
    X = pd.DataFrame(
        {
            "feat": np.arange(n, dtype=float),
            "date": pd.date_range("2026-12-01", periods=n, freq="D"),
        }
    )
    y = np.arange(n, dtype=float)
    pipe = _datetime_pipeline()

    if report_type == "estimator":
        split = int(n * 0.8)
        report = EstimatorReport(
            pipe,
            X_train=X.iloc[:split],
            y_train=y[:split],
            X_test=X.iloc[split:],
            y_test=y[split:],
        )
    else:
        report = evaluate(pipe, X, y, splitter=TimeSeriesSplit(n_splits=5))
    assert CheckTrainTestTimeOverlap().check_function(report) is None


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_when_no_datetime_column(report_type, regression_data):
    """SKD013 needs at least one datetime column to check for overlap."""
    X, y = regression_data
    X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    report = evaluate(
        LinearRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    with pytest.raises(CheckNotApplicable, match="No datetime column found."):
        CheckTrainTestTimeOverlap().check_function(report)


@pytest.mark.filterwarnings("ignore:X does not have valid feature names:UserWarning")
def test_not_applicable_when_x_test_is_not_a_dataframe():
    """SKD013 checks X_test's type independently of X_train's."""
    rng = np.random.default_rng(0)
    X_train_df = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    X_test_arr = rng.normal(size=(20, 3))
    y_train = rng.integers(0, 2, size=40)
    y_test = rng.integers(0, 2, size=20)
    report = EstimatorReport(
        LogisticRegression(),
        X_train=X_train_df,
        y_train=y_train,
        X_test=X_test_arr,
        y_test=y_test,
    )
    with pytest.raises(CheckNotApplicable, match="Got ndarray"):
        CheckTrainTestTimeOverlap().check_function(report)


def test_not_applicable_when_x_train_is_none():
    """SKD013 raises CheckNotApplicable when X_train is unavailable."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = rng.normal(size=100)
    estimator = LinearRegression().fit(X[:80], y[:80])
    report = EstimatorReport(estimator, X_test=X[80:], y_test=y[80:])
    with pytest.raises(CheckNotApplicable, match="Train data is unavailable"):
        CheckTrainTestTimeOverlap().check_function(report)
