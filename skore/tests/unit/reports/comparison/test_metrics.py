"""
Common test for the metrics accessor of a ComparisonReport.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
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


class TestUnsupportedMetric:
    """Test the behaviour of ComparisonReport metrics when some or none of the compared
    reports support the requested metric.

    Originates from <https://github.com/probabl-ai/skore/issues/1473>
    """

    def test_no_report_supports_metric(self, binary_classification_train_test_split):
        """If you call Brier score and none of the sub-reports support it,
        you should get an AttributeError."""
        X_train, X_test, y_train, y_test = binary_classification_train_test_split
        estimator_report_1 = EstimatorReport(
            LinearSVC(),  # Does not support Brier score
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        estimator_report_2 = EstimatorReport(
            LinearSVC(),  # Does not support Brier score
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        report = ComparisonReport([estimator_report_1, estimator_report_2])
        with pytest.raises(AttributeError):
            report.metrics.brier_score()

    def test_some_reports_support_metric(self, binary_classification_train_test_split):
        """If you call `brier_score` and some of the sub-reports support it,
        you should get a dataframe with NaN"""
        X_train, X_test, y_train, y_test = binary_classification_train_test_split
        estimator_report_1 = EstimatorReport(
            DummyClassifier(strategy="uniform", random_state=0),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        estimator_report_2 = EstimatorReport(
            LinearSVC(),  # Does not support Brier score
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        report = ComparisonReport([estimator_report_1, estimator_report_2])
        summary = report.metrics.brier_score()
        assert np.isnan(summary.loc["Brier score"]["LinearSVC"])

    def test_summarize_no_report_supports_metric(
        self, binary_classification_train_test_split
    ):
        """If you call `summarize` with Brier score and none of the sub-reports support
        it, you should get an AttributeError"""
        X_train, X_test, y_train, y_test = binary_classification_train_test_split
        estimator_report_1 = EstimatorReport(
            LinearSVC(),  # Does not support Brier score
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        estimator_report_2 = EstimatorReport(
            LinearSVC(),  # Does not support Brier score
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        report = ComparisonReport([estimator_report_1, estimator_report_2])
        with pytest.raises(ValueError):
            report.metrics.summarize(scoring="brier_score")

    def test_summarize_some_reports_support_metric(
        self, binary_classification_train_test_split
    ):
        """If you call `summarize` with Brier score and some of the sub-reports
        support it, you should get a dataframe with NaN"""
        X_train, X_test, y_train, y_test = binary_classification_train_test_split
        estimator_report_1 = EstimatorReport(
            DummyClassifier(strategy="uniform", random_state=0),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        estimator_report_2 = EstimatorReport(
            LinearSVC(),  # Does not support Brier score
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        report = ComparisonReport([estimator_report_1, estimator_report_2])
        summary = report.metrics.summarize(scoring="brier_score")
        assert np.isnan(summary.frame().loc["Brier score"]["LinearSVC"])


class TestEstimatorReport:
    def test_no_report_supports_metric(self, binary_classification_train_test_split):
        """If you call Brier score and none of the sub-reports support it,
        you should get an AttributeError."""
        X_train, X_test, y_train, y_test = binary_classification_train_test_split
        estimator_report_1 = EstimatorReport(
            LinearSVC(),  # Does not support Brier score
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        estimator_report_2 = EstimatorReport(
            LinearSVC(),  # Does not support Brier score
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        report = ComparisonReport([estimator_report_1, estimator_report_2])
        with pytest.raises(AttributeError):
            report.metrics.brier_score()

    def test_some_reports_support_metric(self, binary_classification_train_test_split):
        """If you call `brier_score` and some of the sub-reports support it,
        you should get a dataframe with NaN"""
        X_train, X_test, y_train, y_test = binary_classification_train_test_split
        estimator_report_1 = EstimatorReport(
            DummyClassifier(strategy="uniform", random_state=0),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        estimator_report_2 = EstimatorReport(
            LinearSVC(),  # Does not support Brier score
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        report = ComparisonReport([estimator_report_1, estimator_report_2])
        summary = report.metrics.brier_score()
        assert np.isnan(summary.loc["Brier score"]["LinearSVC"])

    def test_summarize_no_report_supports_metric(
        self, binary_classification_train_test_split
    ):
        """If you call `summarize` with Brier score and none of the sub-reports support
        it, you should get an AttributeError"""
        X_train, X_test, y_train, y_test = binary_classification_train_test_split
        estimator_report_1 = EstimatorReport(
            LinearSVC(),  # Does not support Brier score
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        estimator_report_2 = EstimatorReport(
            LinearSVC(),  # Does not support Brier score
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        report = ComparisonReport([estimator_report_1, estimator_report_2])
        with pytest.raises(ValueError):
            report.metrics.summarize(scoring="brier_score")

    def test_summarize_some_reports_support_metric(
        self, binary_classification_train_test_split
    ):
        """If you call `summarize` with Brier score and some of the sub-reports
        support it, you should get a dataframe with NaN"""
        X_train, X_test, y_train, y_test = binary_classification_train_test_split
        estimator_report_1 = EstimatorReport(
            DummyClassifier(strategy="uniform", random_state=0),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        estimator_report_2 = EstimatorReport(
            LinearSVC(),  # Does not support Brier score
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        report = ComparisonReport([estimator_report_1, estimator_report_2])
        summary = report.metrics.summarize(scoring="brier_score")
        assert np.isnan(summary.frame().loc["Brier score"]["LinearSVC"])

    def test_cv_summarize_some_reports_support_metric(self, binary_classification_data):
        """If you call `summarize` with Brier score and some of the sub-reports
        support it, you should get a dataframe with NaN"""
        X, y = binary_classification_data
        report_1 = CrossValidationReport(
            DummyClassifier(strategy="uniform", random_state=0), X, y
        )
        report_2 = CrossValidationReport(
            LinearSVC(),  # Does not support Brier score
            X,
            y,
        )
        comp_report = ComparisonReport([report_1, report_2])
        summary = comp_report.metrics.summarize(scoring="brier_score")
        assert np.isnan(summary.frame().loc["Brier score"][("mean", "LinearSVC")])
