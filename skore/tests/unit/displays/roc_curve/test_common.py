"""Common tests for RocCurveDisplay."""

import pytest

from skore import EstimatorReport


def test_frame_columns_consistency(
    logistic_binary_classification_with_train_test,
    logistic_multiclass_classification_with_train_test,
):
    """Check that the frame method returns consistent column names."""
    # binary classification
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    estimator_report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = estimator_report.metrics.roc()

    df = display.frame()
    assert "threshold" in df.columns
    assert "fpr" in df.columns
    assert "tpr" in df.columns

    df_with_auc = display.frame(with_roc_auc=True)
    assert "roc_auc" in df_with_auc.columns

    # multiclass classification
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    estimator_report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = estimator_report.metrics.roc()

    df = display.frame()
    assert "label" in df.columns
    assert "threshold" in df.columns
    assert "fpr" in df.columns
    assert "tpr" in df.columns
