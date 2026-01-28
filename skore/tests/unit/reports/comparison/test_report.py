"""
Tests of ComparisonReport which work regardless whether it holds EstimatorReports or
CrossValidationReports.
"""

import re
from io import BytesIO

import joblib
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from skore import (
    ComparisonReport,
    CrossValidationReport,
    EstimatorReport,
    train_test_split,
)


@pytest.fixture(
    params=[
        "comparison_estimator_reports_binary_classification",
        "comparison_cross_validation_reports_binary_classification",
    ]
)
def report(request):
    return request.getfixturevalue(request.param)


def test_help(capsys, report):
    """Check the help menu works."""
    report.help()

    captured = capsys.readouterr()
    assert "Tools to compare estimators" in captured.out

    # Check that we have a line with accuracy and the arrow associated with it
    assert re.search(
        r"\.accuracy\([^)]*\).*\(↗︎\).*-.*accuracy", captured.out, re.MULTILINE
    )


def test_repr(report):
    """Check the `__repr__` works."""

    assert "ComparisonReport" in repr(report)


def test_metrics_repr(report):
    """Check the repr method of `report.metrics`."""
    repr_str = repr(report.metrics)
    assert "skore.ComparisonReport.metrics" in repr_str
    assert "help()" in repr_str


def test_pickle(tmp_path, report):
    """Check that we can pickle a comparison report."""
    with BytesIO() as stream:
        joblib.dump(report, stream)
        joblib.load(stream)


def test_cross_validation_report_cleaned_up(report):
    """
    When a CrossValidationReport is passed to a ComparisonReport, and computations are
    done on the ComparisonReport, the CrossValidationReport should remain pickle-able.

    Non-regression test for bug found in:
    https://github.com/probabl-ai/skore/pull/1512
    """
    report.metrics.summarize()
    sub_report = next(iter(report.reports_.values()))

    with BytesIO() as stream:
        joblib.dump(sub_report, stream)


def test_metrics_help(capsys, report):
    """Check that the help method writes to the console."""
    report.metrics.help()
    captured = capsys.readouterr()
    assert "Available metrics methods" in captured.out


def test_feature_importance_help(capsys):
    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    reports = {
        name: EstimatorReport(est, X_train=X, X_test=X, y_train=y, y_test=y)
        for name, est in estimators.items()
    }

    comparison_report = ComparisonReport(reports)

    comparison_report.feature_importance.help()
    captured = capsys.readouterr()

    assert "Available feature importance methods" in captured.out
    assert "coefficients" in captured.out

    comparison_report.feature_importance.coefficients().help()
    captured = capsys.readouterr()

    assert "frame" in captured.out
    assert "plot" in captured.out
    assert "set_style" in captured.out


@pytest.mark.parametrize("report", [EstimatorReport, CrossValidationReport])
def test_pos_label_mismatch(report):
    """Check that we raise an error when the positive labels are not the same."""
    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    if report is EstimatorReport:
        reports = {
            name: EstimatorReport(
                est, X_train=X, X_test=X, y_train=y, y_test=y, pos_label=i
            )
            for i, (name, est) in enumerate(estimators.items())
        }
    else:
        reports = {
            name: CrossValidationReport(est, X=X, y=y, pos_label=i)
            for i, (name, est) in enumerate(estimators.items())
        }

    err_msg = "Expected all estimators to have the same positive label."
    with pytest.raises(ValueError, match=err_msg):
        ComparisonReport(reports)


class Test_get_best_report:
    """Test get_best_model."""

    @pytest.fixture
    def classification_estimator_reports(
        self,
    ):
        """Fixture providing classification EstimatorReports for comparison."""
        X, y = make_classification(n_samples=100, random_state=42)
        split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)

        estimator_report_1 = EstimatorReport(
            LogisticRegression(random_state=42), **split_data
        )
        estimator_report_2 = EstimatorReport(
            DecisionTreeClassifier(max_depth=1, random_state=42), **split_data
        )

        return estimator_report_1, estimator_report_2

    @pytest.fixture
    def classification_cv_reports(self):
        X, y = make_classification(n_samples=100, random_state=42)

        report_1 = CrossValidationReport(LogisticRegression(random_state=42), X=X, y=y)
        report_2 = CrossValidationReport(
            DecisionTreeClassifier(max_depth=1, random_state=42), X=X, y=y
        )

        return report_1, report_2

    @pytest.fixture
    def regression_estimator_reports(self):
        X, y = make_regression(n_samples=100, random_state=42)
        split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)

        report_1 = EstimatorReport(LinearRegression(), **split_data)
        report_2 = EstimatorReport(Ridge(alpha=10.0), **split_data)

        return report_1, report_2

    def test_default_metric(self, classification_estimator_reports):
        """Test with EstimatorReport using default metric."""
        report_1, report_2 = classification_estimator_reports
        report = ComparisonReport([report_1, report_2])

        best_report = report.get_best_model()

        assert best_report == report_1

    def test_precision(self, classification_estimator_reports):
        """Test with EstimatorReport using precision."""
        report_1, report_2 = classification_estimator_reports
        report = ComparisonReport([report_1, report_2])

        best_report = report.get_best_model(metric="precision")

        assert best_report == report_1

    def test_precision_with_pos_label(self, classification_estimator_reports):
        """Test with EstimatorReport using precision with pos_label."""
        report_1, report_2 = classification_estimator_reports
        report = ComparisonReport([report_1, report_2])

        best_report = report.get_best_model(metric="precision", pos_label=0)

        assert best_report == report_1

    def test_roc_auc(self, classification_estimator_reports):
        """Test with EstimatorReport using roc_auc metric."""
        report_1, report_2 = classification_estimator_reports
        report = ComparisonReport([report_1, report_2])

        best_report = report.get_best_model(metric="roc_auc")

        assert best_report == report_1

    def test_cross_validation_report_default_metric(self, classification_cv_reports):
        """Test with CrossValidationReport using default metric."""
        cv_report_1, cv_report_2 = classification_cv_reports
        report = ComparisonReport([cv_report_1, cv_report_2])

        best_report = report.get_best_model()

        assert best_report == cv_report_1

    def test_cross_validation_report_custom_metric(self, classification_cv_reports):
        """Test with CrossValidationReport using custom metric."""
        cv_report_1, cv_report_2 = classification_cv_reports
        report = ComparisonReport([cv_report_1, cv_report_2])

        best_report = report.get_best_model(
            metric="precision", metric_kwargs={"average": "macro"}
        )

        assert best_report == cv_report_1

    def test_regression_default_metric(self, regression_estimator_reports):
        """Test with regression task using default metric (r2)."""
        report_1, report_2 = regression_estimator_reports
        report = ComparisonReport([report_1, report_2])

        best_report = report.get_best_model()

        assert best_report == report_1

    def test_lower_is_better_metric(self, regression_estimator_reports):
        """Test with a metric where lower is better (RMSE)."""
        report_1, report_2 = regression_estimator_reports
        report = ComparisonReport([report_1, report_2])

        best_report = report.get_best_model(metric="rmse")

        assert best_report == report_1

    def test_get_best_model_data_source_train(self, regression_estimator_reports):
        """Test using train data source."""
        report_1, report_2 = regression_estimator_reports
        report = ComparisonReport([report_1, report_2])

        best_report = report.get_best_model(data_source="train")

        assert best_report == report_1

    def test_get_best_model_custom_metric_callable(
        self, classification_estimator_reports
    ):
        """Test with a custom metric defined as a callable."""
        report_1, report_2 = classification_estimator_reports
        report = ComparisonReport([report_1, report_2])

        # Define a custom metric callable
        def custom_accuracy(y_true, y_pred):
            """Custom accuracy implementation."""
            return (y_true == y_pred).sum() / len(y_true)

        best_report = report.get_best_model(
            metric=custom_accuracy, response_method="predict"
        )

        assert best_report == report_1
