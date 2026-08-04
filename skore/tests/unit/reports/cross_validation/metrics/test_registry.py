"""Tests for the metrics registry that are specific to ``CrossValidationReport``."""

from sklearn.metrics import accuracy_score, make_scorer


def test_summarize_explicit_custom_metric(
    cross_validation_report_binary_classification,
):
    """``summarize`` exposes the per-split ``split`` column for CV reports."""
    report = cross_validation_report_binary_classification
    report.metrics.add(make_scorer(accuracy_score, response_method="predict"))
    display = report.metrics.summarize(metric="accuracy_score")
    assert set(display.summary["split"]) == {0, 1, 2}
    assert set(display.summary["verbose_name"]) == {"Accuracy Score"}
