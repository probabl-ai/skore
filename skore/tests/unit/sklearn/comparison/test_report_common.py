"""
Tests of ComparisonReport which work regardless whether it holds EstimatorReports or
CrossValidationReports.
"""

import re
from io import BytesIO

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from skore import ComparisonReport, CrossValidationReport, EstimatorReport


@pytest.fixture(params=["report_estimator_reports", "report_cv_reports"])
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
    report.metrics.report_metrics()

    with BytesIO() as stream:
        joblib.dump(report.reports_[0], stream)


def test_metrics_help(capsys, report):
    """Check that the help method writes to the console."""
    report.metrics.help()
    captured = capsys.readouterr()
    assert "Available metrics methods" in captured.out


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
    metrics = comparison_report.metrics.report_metrics(
        pos_label=1, indicator_favorability=True
    )

    assert "Brier score" in metrics.index
    assert "Favorability" in metrics.columns
    assert not metrics["Favorability"].isna().any()
    expected_values = {"(↗︎)", "(↘︎)"}
    actual_values = set(metrics["Favorability"].to_numpy())
    assert actual_values.issubset(expected_values)


@pytest.mark.parametrize("report", [EstimatorReport, CrossValidationReport])
def test_comparison_report_pos_label_mismatch(report):
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


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_comparison_report_display_binary_classification_pos_label(pyplot, metric):
    """Check the behaviour of the display methods when `pos_label` needs to be set."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    report_1 = CrossValidationReport(LogisticRegression(C=1), X, y)
    report_2 = CrossValidationReport(LogisticRegression(C=2), X, y)
    report = ComparisonReport([report_1, report_2])
    with pytest.raises(ValueError, match="pos_label is not specified"):
        getattr(report.metrics, metric)()

    report_1 = CrossValidationReport(LogisticRegression(C=1), X, y, pos_label="A")
    report_2 = CrossValidationReport(LogisticRegression(C=2), X, y, pos_label="A")
    report = ComparisonReport([report_1, report_2])
    display = getattr(report.metrics, metric)()
    display.plot()
    assert "Positive label: A" in display.ax_.get_xlabel()

    display = getattr(report.metrics, metric)(pos_label="B")
    display.plot()
    assert "Positive label: B" in display.ax_.get_xlabel()


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_comparison_report_report_metrics_pos_label_default(metric):
    """Check the default behaviour of `pos_label` in `report_metrics`."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]

    report_1 = CrossValidationReport(LogisticRegression(), X, y)
    report_2 = CrossValidationReport(LogisticRegression(), X, y)
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result_both_labels = report.metrics.report_metrics(scoring=metric).reset_index()
    assert result_both_labels["Label / Average"].to_list() == ["A", "B"]
    result_both_labels = result_both_labels.set_index(["Metric", "Label / Average"])


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_comparison_report_report_metrics_pos_label_overwrite(metric):
    """Check that `pos_label` can be overwritten in `report_metrics`"""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]

    report_1 = CrossValidationReport(LogisticRegression(), X, y, pos_label="B")
    report_2 = CrossValidationReport(LogisticRegression(), X, y, pos_label="B")
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})

    result_both_labels = report.metrics.report_metrics(scoring=metric, pos_label=None)

    result = report.metrics.report_metrics(scoring=metric).reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    for report_name in report.report_names_:
        assert (
            result.loc[metric.capitalize(), ("mean", report_name)]
            == result_both_labels.loc[(metric.capitalize(), "B"), ("mean", report_name)]
        )

    result = report.metrics.report_metrics(scoring=metric, pos_label="A").reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    for report_name in report.report_names_:
        assert (
            result.loc[metric.capitalize(), ("mean", report_name)]
            == result_both_labels.loc[(metric.capitalize(), "A"), ("mean", report_name)]
        )


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_comparison_report_precision_recall_pos_label_default(metric):
    """Check the default behaviour of `pos_label` in `report_metrics`."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    report_1 = CrossValidationReport(LogisticRegression(), X, y)
    report_2 = CrossValidationReport(LogisticRegression(), X, y)
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result_both_labels = getattr(report.metrics, metric)().reset_index()
    assert result_both_labels["Label / Average"].to_list() == ["A", "B"]


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_comparison_report_precision_recall_pos_label_overwrite(metric):
    """Check that `pos_label` can be overwritten in `report_metrics`."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    report_1 = CrossValidationReport(LogisticRegression(), X, y)
    report_2 = CrossValidationReport(LogisticRegression(), X, y)
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})

    result_both_labels = getattr(report.metrics, metric)(pos_label=None)

    result = getattr(report.metrics, metric)(pos_label="B").reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    for report_name in report.report_names_:
        assert (
            result.loc[metric.capitalize(), ("mean", report_name)]
            == result_both_labels.loc[(metric.capitalize(), "B"), ("mean", report_name)]
        )

    result = getattr(report.metrics, metric)(pos_label="A").reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    for report_name in report.report_names_:
        assert (
            result.loc[metric.capitalize(), ("mean", report_name)]
            == result_both_labels.loc[(metric.capitalize(), "A"), ("mean", report_name)]
        )
