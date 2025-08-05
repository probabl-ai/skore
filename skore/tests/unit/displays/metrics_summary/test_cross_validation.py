import numpy as np
import pandas as pd
from skore import CrossValidationReport, MetricsSummaryDisplay


def test_cross_validation_report_flat_index(forest_binary_classification_data):
    """Check that the index is flattened when `flat_index` is True.

    Since `pos_label` is None, then by default a MultiIndex would be returned.
    Here, we force to have a single-index by passing `flat_index=True`.
    """
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=2)
    result = report.metrics.summarize(flat_index=True)
    assert isinstance(result, MetricsSummaryDisplay)
    result_df = result.frame()
    assert result_df.shape == (8, 2)
    assert isinstance(result_df.index, pd.Index)
    assert result_df.index.tolist() == [
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "brier_score",
        "fit_time_s",
        "predict_time_s",
    ]
    assert result_df.columns.tolist() == [
        "randomforestclassifier_mean",
        "randomforestclassifier_std",
    ]


def test_cross_validation_summarize_data_source_external(
    forest_binary_classification_data,
):
    """Check that the `data_source` parameter works when using external data."""
    estimator, X, y = forest_binary_classification_data
    cv_splitter = 2
    report = CrossValidationReport(estimator, X, y, cv_splitter=cv_splitter)
    result = report.metrics.summarize(
        data_source="X_y", X=X, y=y, aggregate=None
    ).frame()
    for split_idx in range(cv_splitter):
        # check that it is equivalent to call the individual estimator report
        report_result = (
            report.estimator_reports_[split_idx]
            .metrics.summarize(data_source="X_y", X=X, y=y)
            .frame()
        )
        np.testing.assert_allclose(
            report_result.iloc[:, 0].to_numpy(), result.iloc[:, split_idx].to_numpy()
        )
