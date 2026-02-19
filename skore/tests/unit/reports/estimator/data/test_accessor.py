import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from skore import EstimatorReport


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"data_source": "invalid"}, "'data_source' options are"),
        ({"subsample_strategy": "invalid"}, "'subsample_strategy' options are"),
    ],
)
def test_analyze_error(forest_binary_classification_with_test, params, err_msg):
    """Check that the `analyze` method raises an error when the data source is not
    valid."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)
    with pytest.raises(ValueError, match=err_msg):
        report.data.analyze(**params)


def test_analyze_data_source_not_available(
    forest_binary_classification_with_test,
):
    """Check that we raise a proper error message when the data source requested is
    not available."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier)

    err_msg = "X_train is required when `data_source='train'`"
    with pytest.raises(ValueError, match=err_msg):
        report.data.analyze(data_source="train")

    err_msg = "X_test is required when `data_source='test'`"
    with pytest.raises(ValueError, match=err_msg):
        report.data.analyze(data_source="test")

    err_msg = "X_train is required when `data_source='both'`"
    with pytest.raises(ValueError, match=err_msg):
        report.data.analyze(data_source="both")

    # if not requesting `y`, we should not raise an error
    report = EstimatorReport(classifier, X_test=X_test)
    display = report.data.analyze(data_source="test", with_y=False)
    np.testing.assert_array_equal(display.summary["dataframe"].to_numpy(), X_test)

    err_msg = "y_test is required when `data_source='test'`"
    with pytest.raises(ValueError, match=err_msg):
        report.data.analyze(data_source="test", with_y=True)


def test_analyze_data_source_without_y():
    """Check the behaviour of `data_source` parameter without including `y`."""
    X, y = make_classification(n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f"Column {i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="Classification target")
    X_train, X_test = X.iloc[:100], X.iloc[100:]
    y_train, y_test = y.iloc[:100], y.iloc[100:]
    classifier = LogisticRegression()

    report = EstimatorReport(
        classifier, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.data.analyze(data_source="train", with_y=False)
    pd.testing.assert_frame_equal(display.summary["dataframe"], X_train)

    display = report.data.analyze(data_source="test", with_y=False)
    pd.testing.assert_frame_equal(display.summary["dataframe"], X_test)

    display = report.data.analyze(data_source="both", with_y=False)
    pd.testing.assert_frame_equal(display.summary["dataframe"], X)


def test_analyze_data_source_with_y():
    """Check the behaviour of `data_source` parameter with `y`."""
    X, y = make_classification(n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f"Column {i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="Classification target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, train_size=50
    )
    classifier = LogisticRegression()

    report = EstimatorReport(
        classifier, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.data.analyze(data_source="train")
    pd.testing.assert_frame_equal(
        display.summary["dataframe"], pd.concat([X_train, y_train], axis=1)
    )

    display = report.data.analyze(data_source="test")
    pd.testing.assert_frame_equal(
        display.summary["dataframe"], pd.concat([X_test, y_test], axis=1)
    )

    display = report.data.analyze(data_source="both")
    pd.testing.assert_frame_equal(
        display.summary["dataframe"], pd.concat([X, y], axis=1)
    )


@pytest.mark.parametrize(
    "X", [[[0, 1], [2, 3]], ((0, 1), (2, 3)), [[0, np.nan], [np.nan, 1]]]
)
@pytest.mark.parametrize("y", [[0, 1], (0, 1)])
def test_analyze_sequence(X, y):
    """Check that lists/tuples are supported for X and y (like in scikit-learn)"""

    report = EstimatorReport(DecisionTreeRegressor(), X_train=X, y_train=y)
    report.data.analyze(data_source="train")  # should not crash


@pytest.mark.parametrize("data_source", ["train", "test", "both"])
@pytest.mark.parametrize(
    "n_targets, target_column_names", [(1, ["Target"]), (2, ["Target 0", "Target 1"])]
)
def test_analyze_numpy_array(data_source, n_targets, target_column_names):
    """Check that NumPy arrays are converted to pandas DataFrames when data are
    retrieved."""
    X, y = make_regression(
        n_samples=100, n_features=2, n_targets=n_targets, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, train_size=50
    )
    classifier = LinearRegression()

    report = EstimatorReport(
        classifier, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.data.analyze(data_source=data_source, with_y=False)
    assert display.summary["dataframe"].columns.to_list() == [
        f"Feature {i}" for i in range(X_train.shape[1])
    ]

    display = report.data.analyze(data_source=data_source, with_y=True)
    assert (
        display.summary["dataframe"].columns.to_list()
        == [f"Feature {i}" for i in range(X_train.shape[1])] + target_column_names
    )


@pytest.mark.parametrize("subsample_strategy", ["head", "random"])
def test_analyze_subsampling(
    forest_binary_classification_with_test, subsample_strategy
):
    """Check that the `subsample` parameter is handled correctly."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    X_test = pd.DataFrame(
        X_test, columns=[f"Feature {i}" for i in range(X_test.shape[1])]
    )
    y_test = pd.Series(y_test, name="Target")
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    display = report.data.analyze(
        data_source="test", subsample=10, subsample_strategy=subsample_strategy, seed=42
    )
    assert display.summary["dataframe"].shape[0] == 10

    if subsample_strategy == "head":
        assert display.summary["dataframe"].index.to_list() == list(range(10))
    else:
        assert display.summary["dataframe"].index.to_list() != list(range(10))
