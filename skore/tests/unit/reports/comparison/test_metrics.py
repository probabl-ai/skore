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
def test_favorability_undefined_metrics(report):
    """Check that we don't introduce NaN when favorability is computed when
    for some estimators, the metric is undefined.

    Non-regression test for:
    https://github.com/probabl-ai/skore/issues/1755
    """

    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    if report is EstimatorReport:
        reports = {
            name: EstimatorReport(
                est, X_train=X, X_test=X, y_train=y, y_test=y, pos_label=1
            )
            for name, est in estimators.items()
        }
    else:
        reports = {
            name: CrossValidationReport(est, X=X, y=y, pos_label=1)
            for name, est in estimators.items()
        }

    comparison_report = ComparisonReport(reports)
    metrics = comparison_report.metrics.summarize()
    assert isinstance(metrics, MetricsSummaryDisplay)
    metrics_df = metrics.frame(favorability=True)

    assert "Brier score" in metrics_df.index
    assert "Favorability" in metrics_df.columns
    assert not metrics_df["Favorability"].isna().any()
    expected_values = {"(↗︎)", "(↘︎)"}
    actual_values = set(metrics_df["Favorability"].to_numpy())
    assert actual_values.issubset(expected_values)


@pytest.mark.parametrize("report", [EstimatorReport, CrossValidationReport])
def test_available_union(report):
    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    if report is EstimatorReport:
        reports = {
            name: EstimatorReport(
                est, X_train=X, X_test=X, y_train=y, y_test=y, pos_label=1
            )
            for name, est in estimators.items()
        }
    else:
        reports = {
            name: CrossValidationReport(est, X=X, y=y, pos_label=1)
            for name, est in estimators.items()
        }

    comparison_report = ComparisonReport(reports)
    available = comparison_report.metrics.available()
    expected = list(
        dict.fromkeys(
            metric
            for sub_report in comparison_report.reports_.values()
            for metric in sub_report.metrics.available()
        )
    )

    assert available == expected


@pytest.mark.parametrize("report", [EstimatorReport, CrossValidationReport])
def test_available_for_single_report_name(report):
    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    if report is EstimatorReport:
        reports = {
            name: EstimatorReport(
                est, X_train=X, X_test=X, y_train=y, y_test=y, pos_label=1
            )
            for name, est in estimators.items()
        }
    else:
        reports = {
            name: CrossValidationReport(est, X=X, y=y, pos_label=1)
            for name, est in estimators.items()
        }

    comparison_report = ComparisonReport(reports)
    assert (
        comparison_report.metrics.available("LinearSVC")
        == reports["LinearSVC"].metrics.available()
    )


@pytest.mark.parametrize("report", [EstimatorReport, CrossValidationReport])
def test_available_with_unknown_report_name_raises(report):
    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    if report is EstimatorReport:
        reports = {
            name: EstimatorReport(
                est, X_train=X, X_test=X, y_train=y, y_test=y, pos_label=1
            )
            for name, est in estimators.items()
        }
    else:
        reports = {
            name: CrossValidationReport(est, X=X, y=y, pos_label=1)
            for name, est in estimators.items()
        }

    comparison_report = ComparisonReport(reports)
    with pytest.raises(ValueError, match="Unknown report name"):
        comparison_report.metrics.available("unknown")
