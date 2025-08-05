import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    make_scorer,
    median_absolute_error,
    r2_score,
)
from sklearn.svm import SVC
from skore import CrossValidationReport, MetricsSummaryDisplay
from skore._sklearn._plot import MetricsSummaryDisplay
from skore._utils._testing import MockEstimator



def test_cross_validation_report_metrics_help(capsys, forest_binary_classification_data):
    """Check that the help method writes to the console."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, cv_splitter=2)

    report.metrics.help()
    captured = capsys.readouterr()
    assert "Available metrics methods" in captured.out


# def test_cross_validation_report_metrics_repr(binary_classification_data):
#     """Check that __repr__ returns a string starting with the expected prefix."""
#     estimator, X, y = binary_classification_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)

#     repr_str = repr(report.metrics)
#     assert "skore.CrossValidationReport.metrics" in repr_str
#     assert "help()" in repr_str


# def _normalize_metric_name(index):
#     """Helper to normalize the metric name present in a pandas index that could be
#     a multi-index or single-index."""
#     # if we have a multi-index, then the metric name is on level 0
#     s = index[0] if isinstance(index, tuple) else index
#     # Remove spaces and underscores
#     return re.sub(r"[^a-zA-Z]", "", s.lower())


# def _check_results_single_metric(report, metric, expected_n_splits, expected_nb_stats):
#     assert hasattr(report.metrics, metric)
#     result = getattr(report.metrics, metric)(aggregate=None)
#     assert isinstance(result, pd.DataFrame)
#     assert result.shape[1] == expected_n_splits
#     # check that we hit the cache
#     result_with_cache = getattr(report.metrics, metric)(aggregate=None)
#     pd.testing.assert_frame_equal(result, result_with_cache)

#     # check that the columns contains the expected split names
#     split_names = result.columns.get_level_values(1).unique()
#     expected_split_names = [f"Split #{i}" for i in range(expected_n_splits)]
#     assert list(split_names) == expected_split_names

#     # check that something was written to the cache
#     assert report._cache != {}
#     report.clear_cache()

#     _check_metrics_names(result, [metric], expected_nb_stats)

#     # check the aggregate parameter
#     stats = ["mean", "std"]
#     result = getattr(report.metrics, metric)(aggregate=stats)
#     # check that the columns contains the expected split names
#     split_names = result.columns.get_level_values(1).unique()
#     assert list(split_names) == stats

#     stats = "mean"
#     result = getattr(report.metrics, metric)(aggregate=stats)
#     # check that the columns contains the expected split names
#     split_names = result.columns.get_level_values(1).unique()
#     assert list(split_names) == [stats]


# def _check_results_summarize(
#     report, params, expected_n_splits, expected_metrics, expected_nb_stats
# ):
#     result = report.metrics.summarize(**params)
#     assert isinstance(result, MetricsSummaryDisplay)
#     result_df = result.frame()
#     assert isinstance(result_df, pd.DataFrame)
#     assert "Favorability" not in result_df.columns
#     assert result_df.shape[1] == expected_n_splits
#     # check that we hit the cache
#     result_with_cache = report.metrics.summarize(**params).frame()
#     pd.testing.assert_frame_equal(result_df, result_with_cache)

#     # check that the columns contains the expected split names
#     split_names = result_df.columns.get_level_values(1).unique()
#     # expected_split_names = [f"Split #{i}" for i in range(expected_n_splits)]
#     expected_split_names = ["mean", "std"]
#     assert list(split_names) == expected_split_names

#     _check_metrics_names(result_df, expected_metrics, expected_nb_stats)

#     # check the aggregate parameter
#     stats = ["mean", "std"]
#     result = report.metrics.summarize(aggregate=stats, **params).frame()
#     # check that the columns contains the expected split names
#     split_names = result.columns.get_level_values(1).unique()
#     assert list(split_names) == stats

#     stats = "mean"
#     result = report.metrics.summarize(aggregate=stats, **params).frame()
#     # check that the columns contains the expected split names
#     split_names = result.columns.get_level_values(1).unique()
#     assert list(split_names) == [stats]


# def _check_metrics_names(result, expected_metrics, expected_nb_stats):
#     assert isinstance(result, pd.DataFrame)
#     assert len(result.index) == expected_nb_stats

#     normalized_expected = {
#         _normalize_metric_name(metric) for metric in expected_metrics
#     }
#     for idx in result.index:
#         normalized_idx = _normalize_metric_name(idx)
#         matches = [metric for metric in normalized_expected if metric == normalized_idx]
#         assert len(matches) == 1, (
#             f"No match found for index '{idx}' in expected metrics:  {expected_metrics}"
#         )


# @pytest.mark.parametrize(
#     "metric, nb_stats",
#     [
#         ("accuracy", 1),
#         ("precision", 2),
#         ("recall", 2),
#         ("brier_score", 1),
#         ("roc_auc", 1),
#         ("log_loss", 1),
#     ],
# )
# def test_cross_validation_report_metrics_binary_classification(
#     binary_classification_data, metric, nb_stats
# ):
#     """Check the behaviour of the metrics methods available for binary
#     classification.
#     """
#     (estimator, X, y), cv = binary_classification_data, 2
#     report = CrossValidationReport(estimator, X, y, cv_splitter=cv)
#     _check_results_single_metric(report, metric, cv, nb_stats)


# @pytest.mark.parametrize(
#     "metric, nb_stats",
#     [
#         ("accuracy", 1),
#         ("precision", 3),
#         ("recall", 3),
#         ("roc_auc", 3),
#         ("log_loss", 1),
#     ],
# )
# def test_cross_validation_report_metrics_multiclass_classification(
#     multiclass_classification_data, metric, nb_stats
# ):
#     """Check the behaviour of the metrics methods available for multiclass
#     classification.
#     """
#     (estimator, X, y), cv = multiclass_classification_data, 2
#     report = CrossValidationReport(estimator, X, y, cv_splitter=cv)
#     _check_results_single_metric(report, metric, cv, nb_stats)


# @pytest.mark.parametrize("metric, nb_stats", [("r2", 1), ("rmse", 1)])
# def test_cross_validation_report_metrics_regression(regression_data, metric, nb_stats):
#     """Check the behaviour of the metrics methods available for regression."""
#     (estimator, X, y), cv = regression_data, 2
#     report = CrossValidationReport(estimator, X, y, cv_splitter=cv)
#     _check_results_single_metric(report, metric, cv, nb_stats)


# @pytest.mark.parametrize("metric, nb_stats", [("r2", 2), ("rmse", 2)])
# def test_cross_validation_report_metrics_regression_multioutput(
#     regression_multioutput_data, metric, nb_stats
# ):
#     """Check the behaviour of the metrics methods available for regression."""
#     (estimator, X, y), cv = regression_multioutput_data, 2
#     report = CrossValidationReport(estimator, X, y, cv_splitter=cv)
#     _check_results_single_metric(report, metric, cv, nb_stats)


# @pytest.mark.parametrize(
#     "scoring, scoring_kwargs",
#     [
#         ("accuracy", None),
#         ("neg_log_loss", None),
#         (accuracy_score, {"response_method": "predict"}),
#         (get_scorer("accuracy"), None),
#     ],
# )
# def test_cross_validation_report_summarize_scoring_single_list_equivalence(
#     binary_classification_data, scoring, scoring_kwargs
# ):
#     """Check that passing a single string, callable, scorer is equivalent to passing a
#     list with a single element."""
#     (estimator, X, y), cv = binary_classification_data, 2
#     report = CrossValidationReport(estimator, X, y, cv_splitter=cv)
#     result_single = report.metrics.summarize(
#         scoring=scoring, scoring_kwargs=scoring_kwargs
#     ).frame()
#     result_list = report.metrics.summarize(
#         scoring=[scoring], scoring_kwargs=scoring_kwargs
#     ).frame()
#     assert result_single.equals(result_list)


# @pytest.mark.parametrize("pos_label, nb_stats", [(None, 2), (1, 1)])
# def test_cross_validation_report_summarize_binary(
#     binary_classification_data,
#     binary_classification_data_svc,
#     pos_label,
#     nb_stats,
# ):
#     """Check the behaviour of the `summarize` method with binary
#     classification. We test both with an SVC that does not support `predict_proba` and a
#     RandomForestClassifier that does.
#     """
#     estimator, X, y = binary_classification_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)
#     expected_metrics = (
#         "precision",
#         "recall",
#         "roc_auc",
#         "brier_score",
#         "fit_time_s",
#         "predict_time_s",
#     )
#     # depending on `pos_label`, we report a stats for each class or not for
#     # precision and recall
#     expected_nb_stats = 2 * nb_stats + 4
#     _check_results_summarize(
#         report,
#         params={"pos_label": pos_label},
#         expected_n_splits=2,
#         expected_metrics=expected_metrics,
#         expected_nb_stats=expected_nb_stats,
#     )

#     # Repeat the same experiment where we the target labels are not [0, 1] but
#     # ["neg", "pos"]. We check that we don't get any error.
#     target_names = np.array(["neg", "pos"], dtype=object)
#     pos_label_name = target_names[pos_label] if pos_label is not None else pos_label
#     y = target_names[y]
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)
#     expected_metrics = (
#         "precision",
#         "recall",
#         "roc_auc",
#         "brier_score",
#         "fit_time_s",
#         "predict_time_s",
#     )
#     # depending on `pos_label`, we report a stats for each class or not for
#     # precision and recall
#     expected_nb_stats = 2 * nb_stats + 4
#     _check_results_summarize(
#         report,
#         params={"pos_label": pos_label_name},
#         expected_n_splits=2,
#         expected_metrics=expected_metrics,
#         expected_nb_stats=expected_nb_stats,
#     )

#     estimator, X, y = binary_classification_data_svc
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)
#     expected_metrics = (
#         "precision",
#         "recall",
#         "roc_auc",
#         "fit_time_s",
#         "predict_time_s",
#     )
#     # depending on `pos_label`, we report a stats for each class or not for
#     # precision and recall
#     expected_nb_stats = 2 * nb_stats + 3
#     _check_results_summarize(
#         report,
#         params={"pos_label": pos_label},
#         expected_n_splits=2,
#         expected_metrics=expected_metrics,
#         expected_nb_stats=expected_nb_stats,
#     )


# def test_cross_validation_report_summarize_multiclass(
#     multiclass_classification_data, multiclass_classification_data_svc
# ):
#     """Check the behaviour of the `summarize` method with multiclass
#     classification.
#     """
#     estimator, X, y = multiclass_classification_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)
#     expected_metrics = (
#         "precision",
#         "recall",
#         "roc_auc",
#         "log_loss",
#         "fit_time_s",
#         "predict_time_s",
#     )
#     # since we are not averaging by default, we report 3 statistics for
#     # precision, recall and roc_auc
#     expected_nb_stats = 3 * 3 + 3
#     _check_results_summarize(
#         report,
#         params={},
#         expected_n_splits=2,
#         expected_metrics=expected_metrics,
#         expected_nb_stats=expected_nb_stats,
#     )

#     estimator, X, y = multiclass_classification_data_svc
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)
#     expected_metrics = ("precision", "recall", "fit_time_s", "predict_time_s")
#     # since we are not averaging by default, we report 3 statistics for
#     # precision and recall
#     expected_nb_stats = 3 * 2 + 2
#     _check_results_summarize(
#         report,
#         params={},
#         expected_n_splits=2,
#         expected_metrics=expected_metrics,
#         expected_nb_stats=expected_nb_stats,
#     )


# def test_cross_validation_report_summarize_regression(regression_data):
#     """Check the behaviour of the `summarize` method with regression."""
#     estimator, X, y = regression_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)
#     expected_metrics = ("r2", "rmse", "fit_time_s", "predict_time_s")
#     _check_results_summarize(
#         report,
#         params={},
#         expected_n_splits=2,
#         expected_metrics=expected_metrics,
#         expected_nb_stats=len(expected_metrics),
#     )


# def test_cross_validation_report_summarize_scoring_kwargs_regression(
#     regression_multioutput_data,
# ):
#     """Check the behaviour of the `summarize` method with scoring kwargs."""
#     estimator, X, y = regression_multioutput_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)
#     assert hasattr(report.metrics, "summarize")
#     result = report.metrics.summarize(
#         scoring_kwargs={"multioutput": "raw_values"}
#     ).frame()
#     assert result.shape == (6, 2)
#     assert isinstance(result.index, pd.MultiIndex)
#     assert result.index.names == ["Metric", "Output"]


# def test_cross_validation_report_summarize_scoring_kwargs_multi_class(
#     multiclass_classification_data,
# ):
#     """Check the behaviour of the `summarize` method with scoring kwargs."""
#     estimator, X, y = multiclass_classification_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)
#     assert hasattr(report.metrics, "summarize")
#     result = report.metrics.summarize(scoring_kwargs={"average": None}).frame()
#     assert result.shape == (12, 2)
#     assert isinstance(result.index, pd.MultiIndex)
#     assert result.index.names == ["Metric", "Label / Average"]


# @pytest.mark.parametrize(
#     "fixture_name, scoring_names, expected_index",
#     [
#         (
#             "regression_data",
#             ["R2", "RMSE", "FIT_TIME", "PREDICT_TIME"],
#             ["R2", "RMSE", "FIT_TIME", "PREDICT_TIME"],
#         ),
#         (
#             "multiclass_classification_data",
#             ["Precision", "Recall", "ROC AUC", "Log Loss", "Fit Time", "Predict Time"],
#             [
#                 "Precision",
#                 "Precision",
#                 "Precision",
#                 "Recall",
#                 "Recall",
#                 "Recall",
#                 "ROC AUC",
#                 "ROC AUC",
#                 "ROC AUC",
#                 "Log Loss",
#                 "Fit Time",
#                 "Predict Time",
#             ],
#         ),
#     ],
# )
# def test_cross_validation_report_summarize_overwrite_scoring_names(
#     request, fixture_name, scoring_names, expected_index
# ):
#     """Test that we can overwrite the scoring names in summarize."""
#     estimator, X, y = request.getfixturevalue(fixture_name)
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)
#     result = report.metrics.summarize(scoring_names=scoring_names).frame()
#     assert result.shape == (len(expected_index), 2)

#     # Get level 0 names if MultiIndex, otherwise get column names
#     result_index = (
#         result.index.get_level_values(0).tolist()
#         if isinstance(result.index, pd.MultiIndex)
#         else result.index.tolist()
#     )
#     assert result_index == expected_index


# @pytest.mark.parametrize("scoring", ["public_metric", "_private_metric"])
# def test_cross_validation_report_summarize_error_scoring_strings(
#     regression_data, scoring
# ):
#     """Check that we raise an error if a scoring string is not a valid metric."""
#     estimator, X, y = regression_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)
#     err_msg = re.escape(f"Invalid metric: {scoring!r}.")
#     with pytest.raises(ValueError, match=err_msg):
#         report.metrics.summarize(scoring=[scoring])


# def test_cross_validation_report_summarize_with_scorer(regression_data):
#     """Check that we can pass scikit-learn scorer with different parameters to
#     the `summarize` method."""
#     estimator, X, y = regression_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)

#     median_absolute_error_scorer = make_scorer(
#         median_absolute_error, response_method="predict"
#     )

#     result = report.metrics.summarize(
#         scoring=[r2_score, median_absolute_error_scorer],
#         scoring_kwargs={"response_method": "predict"},  # only dispatched to r2_score
#         aggregate=None,
#     ).frame()
#     assert result.shape == (2, 2)

#     expected_result = [
#         [
#             r2_score(est_rep.y_test, est_rep.estimator_.predict(est_rep.X_test)),
#             median_absolute_error(
#                 est_rep.y_test, est_rep.estimator_.predict(est_rep.X_test)
#             ),
#         ]
#         for est_rep in report.estimator_reports_
#     ]
#     np.testing.assert_allclose(
#         result.to_numpy(),
#         np.transpose(expected_result),
#     )


# @pytest.mark.parametrize(
#     "scorer, pos_label",
#     [
#         (
#             make_scorer(
#                 f1_score, response_method="predict", average="macro", pos_label=1
#             ),
#             1,
#         ),
#         (
#             make_scorer(
#                 f1_score, response_method="predict", average="macro", pos_label=1
#             ),
#             None,
#         ),
#         (make_scorer(f1_score, response_method="predict", average="macro"), 1),
#     ],
# )
# def test_cross_validation_report_summarize_with_scorer_binary_classification(
#     binary_classification_data, scorer, pos_label
# ):
#     """Check that we can pass scikit-learn scorer with different parameters to
#     the `summarize` method.

#     We also check that we can pass `pos_label` whether to the scorer or to the
#     `summarize` method or consistently to both.
#     """
#     estimator, X, y = binary_classification_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)

#     result = report.metrics.summarize(
#         scoring=["accuracy", accuracy_score, scorer],
#         scoring_kwargs={"response_method": "predict"},
#     ).frame()
#     assert result.shape == (3, 2)


# def test_cross_validation_report_summarize_with_scorer_pos_label_error(
#     binary_classification_data,
# ):
#     """Check that we raise an error when pos_label is passed both in the scorer and
#     globally conducting to a mismatch."""
#     estimator, X, y = binary_classification_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)

#     f1_scorer = make_scorer(
#         f1_score, response_method="predict", average="macro", pos_label=1
#     )
#     err_msg = re.escape(
#         "`pos_label` is passed both in the scorer and to the `summarize` method."
#     )
#     with pytest.raises(ValueError, match=err_msg):
#         report.metrics.summarize(scoring=[f1_scorer], pos_label=0)


# def test_cross_validation_report_summarize_invalid_metric_type(regression_data):
#     """Check that we raise the expected error message if an invalid metric is passed."""
#     estimator, X, y = regression_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)

#     err_msg = re.escape("Invalid type of metric: <class 'int'> for 1")
#     with pytest.raises(ValueError, match=err_msg):
#         report.metrics.summarize(scoring=[1])


# @pytest.mark.parametrize("aggregate", [None, "mean", ["mean", "std"]])
# def test_cross_validation_report_summarize_indicator_favorability(
#     binary_classification_data, aggregate
# ):
#     """Check that the behaviour of `indicator_favorability` is correct."""
#     estimator, X, y = binary_classification_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)
#     result = report.metrics.summarize(
#         indicator_favorability=True, aggregate=aggregate
#     ).frame()
#     assert "Favorability" in result.columns
#     indicator = result["Favorability"]
#     assert indicator.shape == (8,)
#     assert indicator["Precision"].tolist() == ["(↗︎)", "(↗︎)"]
#     assert indicator["Recall"].tolist() == ["(↗︎)", "(↗︎)"]
#     assert indicator["ROC AUC"].tolist() == ["(↗︎)"]
#     assert indicator["Brier score"].tolist() == ["(↘︎)"]
#     assert indicator["Fit time (s)"].tolist() == ["(↘︎)"]
#     assert indicator["Predict time (s)"].tolist() == ["(↘︎)"]


# def test_cross_validation_report_custom_metric(binary_classification_data):
#     """Check that we can compute a custom metric."""
#     estimator, X, y = binary_classification_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)

#     result = report.metrics.custom_metric(
#         metric_function=accuracy_score,
#         response_method="predict",
#     )
#     assert result.shape == (1, 2)
#     assert result.index == ["Accuracy Score"]


# @pytest.mark.parametrize(
#     "error,error_message",
#     [
#         (ValueError("No more fitting"), "Cross-validation interrupted by an error"),
#         (KeyboardInterrupt(), "Cross-validation interrupted manually"),
#     ],
# )
# @pytest.mark.parametrize("n_jobs", [None, 1, 2])
# def test_cross_validation_report_interrupted(
#     binary_classification_data, capsys, error, error_message, n_jobs
# ):
#     """Check that we can interrupt cross-validation without losing all
#     data."""
#     _, X, y = binary_classification_data

#     estimator = MockEstimator(error=error, n_call=0, fail_after_n_clone=8)
#     report = CrossValidationReport(estimator, X, y, cv_splitter=10, n_jobs=n_jobs)

#     captured = capsys.readouterr()
#     assert all(word in captured.out for word in error_message.split(" "))

#     result = report.metrics.custom_metric(
#         metric_function=accuracy_score,
#         response_method="predict",
#     )
#     assert result.shape == (1, 2)
#     assert result.index == ["Accuracy Score"]


# def test_cross_validation_report_brier_score_requires_probabilities():
#     """Check that the Brier score is not defined for estimator that do not
#     implement `predict_proba`.

#     Non-regression test for:
#     https://github.com/probabl-ai/skore/pull/1471
#     """
#     estimator = SVC()  # SVC does not implement `predict_proba` with default parameters
#     X, y = make_classification(n_classes=2, random_state=42)

#     report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=2)
#     assert not hasattr(report.metrics, "brier_score")


# @pytest.mark.parametrize(
#     "aggregate, expected_columns",
#     [
#         (None, ["Split #0", "Split #1"]),
#         ("mean", ["mean"]),
#         ("std", ["std"]),
#         (["mean", "std"], ["mean", "std"]),
#     ],
# )
# def test_cross_validation_timings(
#     binary_classification_data, aggregate, expected_columns
# ):
#     """Check the general behaviour of the `timings` method."""
#     estimator, X, y = binary_classification_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)
#     timings = report.metrics.timings(aggregate=aggregate)
#     assert isinstance(timings, pd.DataFrame)
#     assert timings.index.tolist() == ["Fit time (s)"]
#     assert timings.columns.tolist() == expected_columns

#     report.get_predictions(data_source="train")
#     timings = report.metrics.timings(aggregate=aggregate)
#     assert isinstance(timings, pd.DataFrame)
#     assert timings.index.tolist() == ["Fit time (s)", "Predict time train (s)"]
#     assert timings.columns.tolist() == expected_columns

#     report.get_predictions(data_source="test")
#     timings = report.metrics.timings(aggregate=aggregate)
#     assert isinstance(timings, pd.DataFrame)
#     assert timings.index.tolist() == [
#         "Fit time (s)",
#         "Predict time train (s)",
#         "Predict time test (s)",
#     ]
#     assert timings.columns.tolist() == expected_columns


# @pytest.mark.parametrize("n_jobs", [None, 1, 2])
# def test_cross_validation_report_failure_all_splits(n_jobs):
#     """Check that we raise an error when no estimators were successfully fitted.
#     during the cross-validation process."""
#     X, y = make_classification(n_samples=100, n_features=10, random_state=42)
#     estimator = MockEstimator(
#         error=ValueError("Intentional failure for testing"), fail_after_n_clone=0
#     )

#     err_msg = "Cross-validation failed: no estimators were successfully fitted"
#     with pytest.raises(RuntimeError, match=err_msg):
#         CrossValidationReport(estimator, X, y, n_jobs=n_jobs)


# def test_cross_validation_timings_flat_index(binary_classification_data):
#     """Check the behaviour of the `timings` method display formatting."""
#     estimator, X, y = binary_classification_data
#     report = CrossValidationReport(estimator, X, y, cv_splitter=2)

#     report.get_predictions(data_source="train")
#     report.get_predictions(data_source="test")

#     results = report.metrics.summarize(flat_index=True).frame()
#     assert results.index.tolist() == [
#         "precision_0",
#         "precision_1",
#         "recall_0",
#         "recall_1",
#         "roc_auc",
#         "brier_score",
#         "fit_time_s",
#         "predict_time_s",
#     ]


# @pytest.mark.parametrize("metric", ["precision", "recall"])
# def test_cross_validation_report_summarize_pos_label_overwrite(
#     metric, binary_classification_data
# ):
#     """Check that `pos_label` can be overwritten in `summarize`"""
#     classifier, X, y = binary_classification_data
#     labels = np.array(["A", "B"], dtype=object)
#     y = labels[y]

#     report = CrossValidationReport(classifier, X, y)
#     result_both_labels = report.metrics.summarize(scoring=metric).frame().reset_index()
#     assert result_both_labels["Label / Average"].to_list() == ["A", "B"]
#     result_both_labels = result_both_labels.set_index(["Metric", "Label / Average"])

#     report = CrossValidationReport(classifier, X, y, pos_label="B")
#     result = report.metrics.summarize(scoring=metric).frame().reset_index()
#     assert "Label / Average" not in result.columns
#     result = result.set_index("Metric")
#     assert (
#         result.loc[metric.capitalize(), (report.estimator_name_, "mean")]
#         == result_both_labels.loc[
#             (metric.capitalize(), "B"), (report.estimator_name_, "mean")
#         ]
#     )

#     result = (
#         report.metrics.summarize(scoring=metric, pos_label="A").frame().reset_index()
#     )
#     assert "Label / Average" not in result.columns
#     result = result.set_index("Metric")
#     assert (
#         result.loc[metric.capitalize(), (report.estimator_name_, "mean")]
#         == result_both_labels.loc[
#             (metric.capitalize(), "A"), (report.estimator_name_, "mean")
#         ]
#     )


# @pytest.mark.parametrize("metric", ["precision", "recall"])
# def test_cross_validation_report_precision_recall_pos_label_overwrite(
#     metric, binary_classification_data
# ):
#     """Check that `pos_label` can be overwritten in `summarize`."""
#     classifier, X, y = binary_classification_data
#     labels = np.array(["A", "B"], dtype=object)
#     y = labels[y]

#     report = CrossValidationReport(classifier, X, y)
#     result_both_labels = getattr(report.metrics, metric)().reset_index()
#     assert result_both_labels["Label / Average"].to_list() == ["A", "B"]
#     result_both_labels = result_both_labels.set_index(["Metric", "Label / Average"])

#     result = getattr(report.metrics, metric)(pos_label="B").reset_index()
#     assert "Label / Average" not in result.columns
#     result = result.set_index("Metric")
#     assert (
#         result.loc[metric.capitalize(), (report.estimator_name_, "mean")]
#         == result_both_labels.loc[
#             (metric.capitalize(), "B"), (report.estimator_name_, "mean")
#         ]
#     )

#     result = getattr(report.metrics, metric)(pos_label="A").reset_index()
#     assert "Label / Average" not in result.columns
#     result = result.set_index("Metric")
#     assert (
#         result.loc[metric.capitalize(), (report.estimator_name_, "mean")]
#         == result_both_labels.loc[
#             (metric.capitalize(), "A"), (report.estimator_name_, "mean")
#         ]
#     )
