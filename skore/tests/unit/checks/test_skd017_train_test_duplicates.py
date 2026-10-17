import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

from skore import EstimatorReport, evaluate
from skore._checks.skd017_train_test_duplicates import CheckTrainTestDuplicates
from skore._checks.utils import CheckNotApplicable


def _frame(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])


def test_duplicates_detected_estimator():
    """Rows shared between X_train and X_test trigger SKD017."""
    X_train = _frame(40)
    # Build X_test with 5 rows copied verbatim from X_train plus 15 fresh rows.
    X_test = pd.concat([X_train.iloc[:5], _frame(15, seed=99)], ignore_index=True)
    y_train = np.arange(len(X_train)) % 2
    y_test = np.arange(len(X_test)) % 2
    report = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    explanation = CheckTrainTestDuplicates().check_function(report)
    assert explanation is not None
    assert "5 test row(s)" in explanation
    assert "25.0%" in explanation


def test_no_duplicates_passes_estimator():
    """Disjoint train/test feature matrices do not trigger SKD017."""
    X_train = _frame(40, seed=0)
    X_test = _frame(20, seed=99)
    y_train = np.arange(len(X_train)) % 2
    y_test = np.arange(len(X_test)) % 2
    report = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert CheckTrainTestDuplicates().check_function(report) is None


def test_duplicates_detected_cross_validation():
    """Exact duplicate rows in the dataset leak across CV folds."""
    base = _frame(50, seed=0)
    # Duplicate the whole dataset so every row has an identical twin; any
    # shuffled split places twins on opposite sides of the fold.
    X = pd.concat([base, base], ignore_index=True)
    y = np.arange(len(X)) % 2
    report = evaluate(
        LogisticRegression(),
        X,
        y,
        splitter=KFold(n_splits=5, shuffle=True, random_state=0),
    )
    explanation = CheckTrainTestDuplicates().check_function(report)
    assert explanation is not None
    assert "test row(s)" in explanation


def test_not_applicable_when_x_train_is_none():
    """SKD017 raises CheckNotApplicable when X_train is unavailable."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = rng.normal(size=100)
    estimator = LinearRegression().fit(X[:80], y[:80])
    report = EstimatorReport(estimator, X_test=X[80:], y_test=y[80:])
    with pytest.raises(CheckNotApplicable, match="Train data is unavailable"):
        CheckTrainTestDuplicates().check_function(report)


@pytest.mark.filterwarnings("ignore:X does not have valid feature names:UserWarning")
def test_not_applicable_when_x_test_is_not_a_dataframe():
    """SKD017 needs narwhals-compatible DataFrames for both train and test."""
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
        CheckTrainTestDuplicates().check_function(report)
