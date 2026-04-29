"""Tests for the metrics registry that are specific to ``CrossValidationReport``.

Behaviors that hold across every report kind live in
``tests/unit/reports/test_metrics_registry_common.py``. This module keeps only
tests that depend on a CV-specific data shape: namely the ``split`` column in
``summarize()`` output.
"""

import pytest
from sklearn.metrics import accuracy_score, make_scorer

from skore import CrossValidationReport


@pytest.fixture
def binary_cv_report(logistic_binary_classification_data):
    """Binary classification CV report with LogisticRegression and pos_label=1."""
    estimator, X, y = logistic_binary_classification_data
    return CrossValidationReport(estimator, X=X, y=y, splitter=2, pos_label=1)


class TestSummarizeIntegration:
    def test_summarize_explicit_custom_metric(self, binary_cv_report):
        """``summarize`` exposes the per-split ``split`` column for CV reports."""
        binary_cv_report.metrics.add(
            make_scorer(accuracy_score, response_method="predict")
        )
        display = binary_cv_report.metrics.summarize(metric="accuracy_score")
        assert set(display.data["split"]) == {0, 1}
        assert set(display.data["metric_verbose_name"]) == {"Accuracy Score"}
