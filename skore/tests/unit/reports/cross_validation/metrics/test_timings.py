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
    assert isinstance(timings, pd.DataFrame)
    assert timings.index.tolist() == ["Fit time (s)"]
    assert timings.columns.tolist() == expected_columns

    report.get_predictions(data_source="train")
    timings = report.metrics.timings(aggregate=aggregate)
    assert isinstance(timings, pd.DataFrame)
    assert timings.index.tolist() == ["Fit time (s)", "Predict time train (s)"]
    assert timings.columns.tolist() == expected_columns

    report.get_predictions(data_source="test")
    timings = report.metrics.timings(aggregate=aggregate)
    assert isinstance(timings, pd.DataFrame)
    assert timings.index.tolist() == [
        "Fit time (s)",
        "Predict time train (s)",
        "Predict time test (s)",
    ]
    assert timings.columns.tolist() == expected_columns


def test_timings_flat_index(forest_binary_classification_data):
    """Check the behaviour of the `timings` method display formatting."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    report.get_predictions(data_source="train")
    report.get_predictions(data_source="test")

    results = report.metrics.summarize(flat_index=True).frame()
    assert results.index.tolist() == [
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "brier_score",
        "fit_time_s",
        "predict_time_s",
    ]
