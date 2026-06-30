import pandas as pd
import pytest

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
