import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from skrub import DatetimeEncoder

from skore import EstimatorReport, evaluate
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.model_checks import CheckTrainTestTimeOverlap


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


def test_train_test_time_overlap():
    """Shuffled split triggers overlap; proper temporal split passes."""
    n = 200
    X = pd.DataFrame(
        {
            "feat": np.arange(n, dtype=float),
            "date": pd.date_range("2026-12-01", periods=n, freq="D"),
        }
    )
    y = np.arange(n, dtype=float)
    pipe = _datetime_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=0
    )
    report = EstimatorReport(
        pipe, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD013" in issues.index
    assert "date" in issues.loc["SKD013", "explanation"]

    split = int(n * 0.8)
    report = EstimatorReport(
        pipe,
        X_train=X.iloc[:split],
        y_train=y[:split],
        X_test=X.iloc[split:],
        y_test=y[split:],
    )
    summary = report.checks.summarize()
    assert "SKD013" not in set(summary.frame(section="issue")["code"])
    assert "SKD013" in set(summary.frame(section="passed")["code"])


def test_train_test_time_no_overlap():
    n = 200
    X = pd.DataFrame(
        {
            "feat": np.arange(n, dtype=float),
            "date": pd.date_range("2026-12-01", periods=n, freq="D"),
        }
    )
    y = np.arange(n, dtype=float)
    pipe = _datetime_pipeline()
    split = int(n * 0.8)
    report = EstimatorReport(
        pipe,
        X_train=X.iloc[:split],
        y_train=y[:split],
        X_test=X.iloc[split:],
        y_test=y[split:],
    )
    summary = report.checks.summarize()
    assert "SKD013" not in set(summary.frame(section="issue")["code"])
    assert "SKD013" in set(summary.frame(section="passed")["code"])


def test_train_test_time_overlap_cv():
    """Shuffled CV triggers overlap; time-series CV passes."""
    n = 200
    X = pd.DataFrame(
        {
            "feat": np.arange(n, dtype=float),
            "date": pd.date_range("2026-12-01", periods=n, freq="D"),
        }
    )
    y = np.arange(n, dtype=float)
    pipe = _datetime_pipeline()

    report = evaluate(
        pipe, X, y, splitter=KFold(n_splits=5, shuffle=True, random_state=0)
    )
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD013" in issues.index
    assert "date" in issues.loc["SKD013", "explanation"]

    report = evaluate(pipe, X, y, splitter=TimeSeriesSplit(n_splits=5))
    summary = report.checks.summarize()
    assert "SKD013" not in set(summary.frame(section="issue")["code"])
    assert "SKD013" in set(summary.frame(section="passed")["code"])


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
