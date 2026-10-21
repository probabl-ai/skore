import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

from skore import EstimatorReport, evaluate
from skore._checks.skd017_train_test_duplicates import CheckTrainTestDuplicates
from skore._checks.utils import CheckNotApplicable


def _frame(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])


def test_duplicates_detected_estimator():
    """Duplicate rows in the re-concatenated dataset trigger SKD017."""
    X_train = _frame(40)
    # 5 test rows are verbatim copies of train rows; after re-concatenation the
    # dataset has 5 duplicate pairs, i.e. 10 rows that have an identical twin.
    X_test = pd.concat([X_train.iloc[:5], _frame(15, seed=99)], ignore_index=True)
    y_train = np.arange(len(X_train)) % 2
    # Labels of the copied test rows match their train twins (both arange % 2).
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
    assert "10 row(s)" in explanation
    # 10 duplicated rows out of 60 total.
    assert "16.7%" in explanation


def test_no_duplicates_passes_estimator():
    """A dataset with no duplicate rows does not trigger SKD017."""
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
    """Duplicate rows in the unsplit dataset are flagged once on the full data."""
    base = _frame(50, seed=0)
    # Duplicate the whole dataset so every row has an identical twin.
    X = pd.concat([base, base], ignore_index=True)
    # Labels of twins match (arange % 2 with an even offset of 50).
    y = np.arange(len(X)) % 2
    report = evaluate(
        LogisticRegression(),
        X,
        y,
        splitter=KFold(n_splits=5, shuffle=True, random_state=0),
    )
    explanation = CheckTrainTestDuplicates().check_function(report)
    assert explanation is not None
    # All 100 rows have an identical twin.
    assert "100 row(s)" in explanation
    assert "100.0%" in explanation


def test_not_applicable_when_x_train_is_none():
    """SKD017 raises CheckNotApplicable when X_train is unavailable."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = rng.normal(size=100)
    estimator = LinearRegression().fit(X[:80], y[:80])
    report = EstimatorReport(estimator, X_test=X[80:], y_test=y[80:])
    with pytest.raises(CheckNotApplicable, match="Train data is unavailable"):
        CheckTrainTestDuplicates().check_function(report)


def test_duplicates_with_nan_are_detected():
    """Duplicate rows containing NaN still count as duplicates."""
    # The NaN-containing train row is copied into the test set, so after
    # re-concatenation it forms a duplicate pair.
    X_train = pd.DataFrame({"a": [1.0, np.nan, 2.0, 3.0], "b": [1.0, 2.0, 3.0, 4.0]})
    X_test = pd.DataFrame({"a": [np.nan, 9.0], "b": [2.0, 9.0]})
    report = EstimatorReport(
        HistGradientBoostingClassifier(),
        X_train=X_train,
        y_train=[0, 1, 0, 1],
        X_test=X_test,
        y_test=[1, 9],  # first test row's label matches its NaN train twin (1)
    )
    explanation = CheckTrainTestDuplicates().check_function(report)
    assert explanation is not None
    assert "2 row(s)" in explanation


def test_identical_features_different_label_not_counted():
    """Rows sharing features but not label are not counted as duplicates."""
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    X_test = X_train.iloc[:1].copy()  # same features as train row 0
    y_train = [0, 1, 0]
    y_test = [1]  # different label from train row 0 (which is 0)
    report = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert CheckTrainTestDuplicates().check_function(report) is None


def test_not_applicable_when_rows_cannot_be_compared():
    """Uncomparable (list-valued) cells degrade to CheckNotApplicable."""
    # A column-dropping transformer lets the estimator fit while the report
    # still stores the raw list-valued column that cannot be de-duplicated.
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
    with pytest.raises(CheckNotApplicable):
        CheckTrainTestDuplicates().check_function(report)


@pytest.mark.filterwarnings("ignore:X does not have valid feature names:UserWarning")
def test_duplicates_detected_with_numpy_arrays():
    """Array inputs are normalized to DataFrames and still checked."""
    X_train = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    X_test = np.array([[1.0, 1.0]])  # verbatim copy of the first train row
    report = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=[0, 1, 0],
        X_test=X_test,
        y_test=[0],  # same label as the matching train row
    )
    explanation = CheckTrainTestDuplicates().check_function(report)
    assert explanation is not None
    assert "2 row(s)" in explanation
