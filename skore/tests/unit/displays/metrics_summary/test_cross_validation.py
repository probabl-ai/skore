import re

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    make_scorer,
    median_absolute_error,
    r2_score,
)

from skore import CrossValidationReport, MetricsSummaryDisplay


def test_flat_index(forest_binary_classification_data):
    """Check that the index is flattened when `flat_index` is True.

    Since `pos_label` is None, then by default a MultiIndex would be returned.
    Here, we force to have a single-index by passing `flat_index=True`.
    """
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    result = report.metrics.summarize(flat_index=True)
    assert isinstance(result, MetricsSummaryDisplay)
    result_df = result.frame()
    assert result_df.shape == (9, 2)
    assert isinstance(result_df.index, pd.Index)
    assert result_df.index.tolist() == [
        "accuracy",
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


def test_data_source_external(
    forest_binary_classification_data,
):
    """Check that the `data_source` parameter works when using external data."""
    estimator, X, y = forest_binary_classification_data
    splitter = 2
    report = CrossValidationReport(estimator, X, y, splitter=splitter)
    result = report.metrics.summarize(
        data_source="X_y", X=X, y=y, aggregate=None
    ).frame()
    for split_idx in range(splitter):
        # check that it is equivalent to call the individual estimator report
        report_result = (
            report.reports_[split_idx]
            .metrics.summarize(data_source="X_y", X=X, y=y)
            .frame()
        )
        np.testing.assert_allclose(
            report_result.iloc[:, 0].to_numpy(), result.iloc[:, split_idx].to_numpy()
        )


def _normalize_metric_name(index):
    """Helper to normalize the metric name present in a pandas index that could be
    a multi-index or single-index."""
    # if we have a multi-index, then the metric name is on level 0
    s = index[0] if isinstance(index, tuple) else index
    # Remove spaces and underscores
    return re.sub(r"[^a-zA-Z]", "", s.lower())


def _check_metrics_names(result, expected_metrics, expected_nb_stats):
    assert isinstance(result, pd.DataFrame)
    assert len(result.index) == expected_nb_stats

    normalized_expected = {
        _normalize_metric_name(metric) for metric in expected_metrics
    }
    for idx in result.index:
        normalized_idx = _normalize_metric_name(idx)
        matches = [metric for metric in normalized_expected if metric == normalized_idx]
        assert len(matches) == 1, (
            f"No match found for index '{idx}' in expected metrics:  {expected_metrics}"
        )


def _check_results_summarize(
    report, params, expected_n_splits, expected_metrics, expected_nb_stats
):
    result = report.metrics.summarize(**params)
    assert isinstance(result, MetricsSummaryDisplay)
    result_df = result.frame()
    assert isinstance(result_df, pd.DataFrame)
    assert "Favorability" not in result_df.columns
    assert result_df.shape[1] == expected_n_splits
    # check that we hit the cache
    result_with_cache = report.metrics.summarize(**params).frame()
    pd.testing.assert_frame_equal(result_df, result_with_cache)

    # check that the columns contains the expected split names
    split_names = result_df.columns.get_level_values(1).unique()
    # expected_split_names = [f"Split #{i}" for i in range(expected_n_splits)]
    expected_split_names = ["mean", "std"]
    assert list(split_names) == expected_split_names

    _check_metrics_names(result_df, expected_metrics, expected_nb_stats)

    # check the aggregate parameter
    stats = ["mean", "std"]
    result = report.metrics.summarize(aggregate=stats, **params).frame()
    # check that the columns contains the expected split names
    split_names = result.columns.get_level_values(1).unique()
    assert list(split_names) == stats

    stats = "mean"
    result = report.metrics.summarize(aggregate=stats, **params).frame()
    # check that the columns contains the expected split names
    split_names = result.columns.get_level_values(1).unique()
    assert list(split_names) == [stats]


@pytest.mark.parametrize(
    "metric, metric_kwargs",
    [
        ("accuracy", None),
        ("neg_log_loss", None),
        (accuracy_score, {"response_method": "predict"}),
        (get_scorer("accuracy"), None),
    ],
)
def test_metric_single_list_equivalence(
    forest_binary_classification_data, metric, metric_kwargs
):
    """Check that passing a single string, callable, scorer is equivalent to passing a
    list with a single element."""
    (estimator, X, y), cv = forest_binary_classification_data, 2
    report = CrossValidationReport(estimator, X, y, splitter=cv)
    result_single = report.metrics.summarize(
        metric=metric, metric_kwargs=metric_kwargs
    ).frame()
    result_list = report.metrics.summarize(
        metric=[metric], metric_kwargs=metric_kwargs
    ).frame()
    assert result_single.equals(result_list)


@pytest.mark.parametrize("pos_label, nb_stats", [(None, 2), (1, 1)])
def test_binary_classification(
    forest_binary_classification_data,
    svc_binary_classification_data,
    pos_label,
    nb_stats,
):
    """Check the behaviour of the `MetricsSummaryDisplay` method with binary
    classification. We test both with an SVC that does not support `predict_proba` and a
    RandomForestClassifier that does.
    """
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    expected_metrics = (
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "brier_score",
        "fit_time_s",
        "predict_time_s",
    )
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 5
    _check_results_summarize(
        report,
        params={"pos_label": pos_label},
        expected_n_splits=2,
        expected_metrics=expected_metrics,
        expected_nb_stats=expected_nb_stats,
    )

    # Repeat the same experiment where we the target labels are not [0, 1] but
    # ["neg", "pos"]. We check that we don't get any error.
    target_names = np.array(["neg", "pos"], dtype=object)
    pos_label_name = target_names[pos_label] if pos_label is not None else pos_label
    y = target_names[y]
    report = CrossValidationReport(estimator, X, y, splitter=2)
    expected_metrics = (
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "brier_score",
        "fit_time_s",
        "predict_time_s",
    )
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 5
    _check_results_summarize(
        report,
        params={"pos_label": pos_label_name},
        expected_n_splits=2,
        expected_metrics=expected_metrics,
        expected_nb_stats=expected_nb_stats,
    )

    estimator, X, y = svc_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    expected_metrics = (
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "fit_time_s",
        "predict_time_s",
    )
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 4
    _check_results_summarize(
        report,
        params={"pos_label": pos_label},
        expected_n_splits=2,
        expected_metrics=expected_metrics,
        expected_nb_stats=expected_nb_stats,
    )


def test_multiclass_classification(
    forest_multiclass_classification_data, svc_multiclass_classification_data
):
    """Check the behaviour of the `MetricsSummaryDisplay` method with multiclass
    classification.
    """
    estimator, X, y = forest_multiclass_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    expected_metrics = (
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "log_loss",
        "fit_time_s",
        "predict_time_s",
    )
    # since we are not averaging by default, we report 3 statistics for
    # precision, recall and roc_auc
    expected_nb_stats = 3 * 3 + 4
    _check_results_summarize(
        report,
        params={},
        expected_n_splits=2,
        expected_metrics=expected_metrics,
        expected_nb_stats=expected_nb_stats,
    )

    estimator, X, y = svc_multiclass_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    expected_metrics = (
        "accuracy",
        "precision",
        "recall",
        "fit_time_s",
        "predict_time_s",
    )
    # since we are not averaging by default, we report 3 statistics for
    # precision and recall
    expected_nb_stats = 3 * 2 + 3
    _check_results_summarize(
        report,
        params={},
        expected_n_splits=2,
        expected_metrics=expected_metrics,
        expected_nb_stats=expected_nb_stats,
    )


def test_regression(linear_regression_data):
    """Check the behaviour of the `MetricsSummaryDisplay` method with regression."""
    estimator, X, y = linear_regression_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    expected_metrics = ("r2", "rmse", "fit_time_s", "predict_time_s")
    _check_results_summarize(
        report,
        params={},
        expected_n_splits=2,
        expected_metrics=expected_metrics,
        expected_nb_stats=len(expected_metrics),
    )


def test_metric_kwargs_regression(
    linear_regression_multioutput_data,
):
    """Check the behaviour of the `MetricsSummaryDisplay` method with scoring kwargs."""
    estimator, X, y = linear_regression_multioutput_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    assert hasattr(report.metrics, "summarize")
    result = report.metrics.summarize(
        metric_kwargs={"multioutput": "raw_values"}
    ).frame()
    assert result.shape == (6, 2)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Metric", "Output"]


def test_metric_kwargs_multi_class(
    forest_multiclass_classification_data,
):
    """Check the behaviour of the `MetricsSummaryDisplay` method with scoring kwargs."""
    estimator, X, y = forest_multiclass_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    assert hasattr(report.metrics, "summarize")
    result = report.metrics.summarize(metric_kwargs={"average": None}).frame()
    assert result.shape == (13, 2)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Metric", "Label / Average"]


@pytest.mark.parametrize(
    "fixture_name, metric, expected_index",
    [
        (
            "linear_regression_data",
            {
                "R2": "r2",
                "RMSE": "rmse",
                "FIT_TIME": "_fit_time",
                "PREDICT_TIME": "_predict_time",
            },
            ["R2", "RMSE", "FIT_TIME", "PREDICT_TIME"],
        ),
        (
            "forest_multiclass_classification_data",
            {
                "Accuracy": "accuracy",
                "Precision": "_precision",
                "Recall": "_recall",
                "ROC AUC": "_roc_auc",
                "Log Loss": "_log_loss",
                "Fit Time": "_fit_time",
                "Predict Time": "_predict_time",
            },
            [
                "Accuracy",
                "Precision",
                "Precision",
                "Precision",
                "Recall",
                "Recall",
                "Recall",
                "ROC AUC",
                "ROC AUC",
                "ROC AUC",
                "Log Loss",
                "Fit Time",
                "Predict Time",
            ],
        ),
    ],
)
def test_overwrite_metric_names(request, fixture_name, metric, expected_index):
    """Test that we can overwrite the scoring names in `MetricsSummaryDisplay`."""
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X, y, splitter=2)
    result = report.metrics.summarize(metric=metric).frame()
    assert result.shape == (len(expected_index), 2)

    # Get level 0 names if MultiIndex, otherwise get column names
    result_index = (
        result.index.get_level_values(0).tolist()
        if isinstance(result.index, pd.MultiIndex)
        else result.index.tolist()
    )
    assert result_index == expected_index


@pytest.mark.parametrize("metric", ["public_metric", "_private_metric"])
def test_error_metric_strings(linear_regression_data, metric):
    """Check that we raise an error if a metric string is not a valid metric."""
    estimator, X, y = linear_regression_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    err_msg = re.escape(f"Invalid metric: {metric!r}.")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[metric])


def test_scorer(linear_regression_data):
    """Check that we can pass scikit-learn scorer with different parameters to
    the `MetricsSummaryDisplay` method."""
    estimator, X, y = linear_regression_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    median_absolute_error_scorer = make_scorer(
        median_absolute_error, response_method="predict"
    )

    result = report.metrics.summarize(
        metric=[r2_score, median_absolute_error_scorer],
        metric_kwargs={"response_method": "predict"},  # only dispatched to r2_score
        aggregate=None,
    ).frame()
    assert result.shape == (2, 2)

    expected_result = [
        [
            r2_score(est_rep.y_test, est_rep.estimator_.predict(est_rep.X_test)),
            median_absolute_error(
                est_rep.y_test, est_rep.estimator_.predict(est_rep.X_test)
            ),
        ]
        for est_rep in report.reports_
    ]
    np.testing.assert_allclose(
        result.to_numpy(),
        np.transpose(expected_result),
    )


@pytest.mark.parametrize(
    "scorer, pos_label",
    [
        (
            make_scorer(
                f1_score, response_method="predict", average="macro", pos_label=1
            ),
            1,
        ),
        (
            make_scorer(
                f1_score, response_method="predict", average="macro", pos_label=1
            ),
            None,
        ),
        (make_scorer(f1_score, response_method="predict", average="macro"), 1),
    ],
)
def test_scorer_binary_classification(
    forest_binary_classification_data, scorer, pos_label
):
    """Check that we can pass scikit-learn scorer with different parameters to
    the `MetricsSummaryDisplay` method.

    We also check that we can pass `pos_label` whether to the scorer or to the
    `MetricsSummaryDisplay` method or consistently to both.
    """
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    result = report.metrics.summarize(
        metric=["accuracy", accuracy_score, scorer],
        metric_kwargs={"response_method": "predict"},
    ).frame()
    assert result.shape == (3, 2)


def test_scorer_pos_label_error(
    forest_binary_classification_data,
):
    """Check that we raise an error when pos_label is passed both in the scorer and
    globally conducting to a mismatch."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    f1_scorer = make_scorer(
        f1_score, response_method="predict", average="macro", pos_label=1
    )
    err_msg = re.escape(
        "`pos_label` is passed both in the scorer and to the `summarize` method."
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[f1_scorer], pos_label=0)


def test_invalid_metric_type(linear_regression_data):
    """Check that we raise the expected error message if an invalid metric is passed."""
    estimator, X, y = linear_regression_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    err_msg = re.escape("Invalid type of metric: <class 'int'> for 1")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[1])


@pytest.mark.parametrize("aggregate", [None, "mean", ["mean", "std"]])
def test_favorability(forest_binary_classification_data, aggregate):
    """Check that the behaviour of `favorability` is correct."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    result = report.metrics.summarize(favorability=True, aggregate=aggregate).frame()
    assert "Favorability" in result.columns
    indicator = result["Favorability"]
    # assert indicator.shape == (9,)
    assert indicator["Accuracy"].tolist() == ["(↗︎)"]
    assert indicator["Precision"].tolist() == ["(↗︎)", "(↗︎)"]
    assert indicator["Recall"].tolist() == ["(↗︎)", "(↗︎)"]
    assert indicator["ROC AUC"].tolist() == ["(↗︎)"]
    assert indicator["Brier score"].tolist() == ["(↘︎)"]
    assert indicator["Fit time (s)"].tolist() == ["(↘︎)"]
    assert indicator["Predict time (s)"].tolist() == ["(↘︎)"]


def test_overwrite_scoring_names_with_dict_cross_validation(
    forest_multiclass_classification_data,
):
    """Test that we can overwrite the scoring names using dict scoring in
    CrossValidationReport."""
    estimator, X, y = forest_multiclass_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    scoring_dict = {
        "Custom Precision": "_precision",
        "Custom Recall": "_recall",
        "Custom ROC AUC": "_roc_auc",
    }

    result = report.metrics.summarize(metric=scoring_dict).frame()

    # Check that custom names are used
    result_index = (
        result.index.get_level_values(0).tolist()
        if isinstance(result.index, pd.MultiIndex)
        else result.index.tolist()
    )

    assert "Custom Precision" in result_index
    assert "Custom Recall" in result_index
    assert "Custom ROC AUC" in result_index
