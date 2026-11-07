import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit

from skore import CrossValidationReport


@pytest.mark.parametrize(
    "aggregate, expected_columns",
    [
        (None, ["Split #0", "Split #1"]),
        ("mean", ["mean"]),
        ("std", ["std"]),
        (["mean", "std"], ["mean", "std"]),
    ],
)
def test_timings(forest_binary_classification_data, aggregate, expected_columns):
    """Check the general behaviour of the `timings` method."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    timings = report.metrics.timings(aggregate=aggregate)
    if len(expected_columns) == 1:
        assert isinstance(timings, pd.Series)
        assert timings.name == expected_columns[0]
    else:
        assert isinstance(timings, pd.DataFrame)
        assert timings.columns.tolist() == expected_columns
    assert timings.index.tolist() == ["Fit time (s)", "Predict time test (s)"]

    report.get_predictions(data_source="train")
    timings = report.metrics.timings(aggregate=aggregate)
    if len(expected_columns) == 1:
        assert isinstance(timings, pd.Series)
        assert timings.name == expected_columns[0]
    else:
        assert isinstance(timings, pd.DataFrame)
        assert timings.columns.tolist() == expected_columns
    assert timings.index.tolist() == [
        "Fit time (s)",
        "Predict time train (s)",
        "Predict time test (s)",
    ]


def test_single_split_std_is_missing(regression_data):
    """A one-split report stores an undefined standard deviation as missing."""
    X, y = regression_data
    report = CrossValidationReport(
        LinearRegression(),
        X,
        y,
        splitter=ShuffleSplit(n_splits=1, random_state=0),
    )
    frame = report.metrics.summarize().frame(flat_index=False)
    std = frame.xs("std", level="aggregate", axis=1)
    assert std.dtypes.iloc[0] == pd.Float64Dtype()
    assert std.isna().all().all()
    pd.testing.assert_series_equal(
        report.metrics.summarize().frame(aggregate="mean"),
        report.metrics.summarize().frame(aggregate=None),
        check_names=False,
    )

    timings = report.metrics.timings()
    assert timings["std"].dtype == pd.Float64Dtype()
    assert timings["std"].isna().all()
