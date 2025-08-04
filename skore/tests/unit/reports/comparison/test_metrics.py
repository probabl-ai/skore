"""
Common test for the metrics accessor of a ComparisonReport.
"""

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from skore import ComparisonReport, CrossValidationReport, EstimatorReport
from skore._sklearn._plot import MetricsSummaryDisplay


@pytest.mark.parametrize("report", [EstimatorReport, CrossValidationReport])
def test_comparison_report_favorability_undefined_metrics(report):
    """Check that we don't introduce NaN when favorability is computed when
    for some estimators, the metric is undefined.

    Non-regression test for:
    https://github.com/probabl-ai/skore/issues/1755
    """

    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    if report is EstimatorReport:
        reports = {
            name: EstimatorReport(est, X_train=X, X_test=X, y_train=y, y_test=y)
            for name, est in estimators.items()
        }
    else:
        reports = {
            name: CrossValidationReport(est, X=X, y=y)
            for name, est in estimators.items()
        }

    comparison_report = ComparisonReport(reports)
    metrics = comparison_report.metrics.summarize(
        pos_label=1, indicator_favorability=True
    )
    assert isinstance(metrics, MetricsSummaryDisplay)
    metrics_df = metrics.frame()

    assert "Brier score" in metrics_df.index
    assert "Favorability" in metrics_df.columns
    assert not metrics_df["Favorability"].isna().any()
    expected_values = {"(↗︎)", "(↘︎)"}
    actual_values = set(metrics_df["Favorability"].to_numpy())
    assert actual_values.issubset(expected_values)
