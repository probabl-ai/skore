import pytest

from skore import (
    ConfusionMatrixDisplay,
    CrossValidationReport,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)


def _iter_estimator_reports(report):
    if hasattr(report, "estimator_reports_"):
        yield from report.estimator_reports_
    elif hasattr(report, "reports_"):
        for child_report in report.reports_.values():
            yield from _iter_estimator_reports(child_report)
    else:
        yield report


@pytest.mark.parametrize(
    ("fixture_name", "display_name", "display_type"),
    [
        (
            "cross_validation_report_binary_classification",
            "roc",
            RocCurveDisplay,
        ),
        (
            "cross_validation_report_binary_classification",
            "precision_recall",
            PrecisionRecallCurveDisplay,
        ),
        (
            "cross_validation_report_binary_classification",
            "confusion_matrix",
            ConfusionMatrixDisplay,
        ),
        (
            "comparison_estimator_reports_binary_classification",
            "roc",
            RocCurveDisplay,
        ),
        (
            "comparison_estimator_reports_binary_classification",
            "precision_recall",
            PrecisionRecallCurveDisplay,
        ),
        (
            "comparison_estimator_reports_binary_classification",
            "confusion_matrix",
            ConfusionMatrixDisplay,
        ),
        (
            "comparison_cross_validation_reports_binary_classification",
            "roc",
            RocCurveDisplay,
        ),
        (
            "comparison_cross_validation_reports_binary_classification",
            "precision_recall",
            PrecisionRecallCurveDisplay,
        ),
        (
            "comparison_cross_validation_reports_binary_classification",
            "confusion_matrix",
            ConfusionMatrixDisplay,
        ),
    ],
)
def test_classification_displays_delegate_to_estimator_reports(
    fixture_name,
    display_name,
    display_type,
    request,
):
    report = request.getfixturevalue(fixture_name)

    getattr(report.metrics, display_name)()

    for estimator_report in _iter_estimator_reports(report):
        assert any(
            isinstance(cached_value, display_type)
            for cached_value in estimator_report._cache.values()
        )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "comparison_estimator_reports_regression",
        "comparison_cross_validation_reports_regression",
    ],
)
def test_prediction_error_delegates_to_estimator_reports(fixture_name, request):
    report = request.getfixturevalue(fixture_name)

    report.metrics.prediction_error(seed=0)

    for estimator_report in _iter_estimator_reports(report):
        assert any(
            isinstance(cached_value, PredictionErrorDisplay)
            for cached_value in estimator_report._cache.values()
        )


def test_cv_prediction_error_delegates_to_estimator_reports(linear_regression_data):
    estimator, X, y = linear_regression_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    report.metrics.prediction_error(seed=0)

    for estimator_report in _iter_estimator_reports(report):
        assert any(
            isinstance(cached_value, PredictionErrorDisplay)
            for cached_value in estimator_report._cache.values()
        )
