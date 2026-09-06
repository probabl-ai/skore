import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression

from skore import evaluate
from skore._checks.skd003_metrics_consistency import (
    CheckMetricsConsistencyAcrossSplits,
    detect_outliers_modified_zscore,
)


def test_passes_when_splits_are_consistent(regression_data):
    """SKD003 does not fire when metrics are consistent across splits."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)
    assert CheckMetricsConsistencyAcrossSplits().check_function(report) is None


def test_detects_inconsistent_splits():
    """Check that the inconsistent performance across splits issue is detected."""
    X, y = make_classification(n_samples=400, n_features=5, random_state=0)
    report = evaluate(LogisticRegression(random_state=0), X, y, splitter=5)
    assert CheckMetricsConsistencyAcrossSplits().check_function(report) is None

    # Corrupt the first split
    y[0 : len(y) // 5] = np.random.RandomState(0).randint(0, 2, len(y) // 5)
    report = evaluate(LogisticRegression(random_state=0), X, y, splitter=5)
    explanation = CheckMetricsConsistencyAcrossSplits().check_function(report)
    assert explanation is not None
    assert "split #0" in explanation
    n_metrics = (
        len(
            report.metrics.summarize(data_source="test").frame(
                aggregate=None, flat_index=True
            )
        )
        - 2  # -2 for the timing metrics
    )
    assert f"for {n_metrics}/{n_metrics} metrics" in explanation


def test_detect_outliers_modified_zscore_flags_extreme_value():
    scores = np.array([1.0, 1.1, 0.9, 1.05, 50.0])
    outliers = detect_outliers_modified_zscore(scores)
    assert outliers.tolist() == [False, False, False, False, True]


def test_detect_outliers_modified_zscore_zero_mad_returns_no_outliers():
    """When all scores are identical, MAD is 0 and nothing is flagged."""
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    outliers = detect_outliers_modified_zscore(scores)
    assert not outliers.any()
