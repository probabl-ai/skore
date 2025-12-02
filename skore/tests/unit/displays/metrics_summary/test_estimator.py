import re

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    make_scorer,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

from skore import EstimatorReport


def _normalize_metric_name(column):
    """Helper to normalize the metric name present in a pandas index that could be
    a multi-index or single-index."""
    # if we have a multi-index, then the metric name is on level 0
    s = column[0] if isinstance(column, tuple) else column
    # Remove spaces and underscores and (s) suffix
    s = s.lower().replace(" (s)", "")
    return re.sub(r"[^a-zA-Z]", "", s)


def _check_results_summarize(result, expected_metrics, expected_nb_stats):
    assert isinstance(result, pd.DataFrame)
    assert len(result.index) == expected_nb_stats

    normalized_expected = {
        _normalize_metric_name(metric) for metric in expected_metrics
    }
    for column in result.index:
        normalized_column = _normalize_metric_name(column)
        matches = [
            metric for metric in normalized_expected if metric == normalized_column
        ]
        assert len(matches) == 1, (
            f"No match found for column '{column}' in expected metrics: "
            f" {expected_metrics}"
        )


@pytest.mark.parametrize("pos_label, nb_stats", [(None, 2), (1, 1)])
@pytest.mark.parametrize("data_source", ["test", "X_y"])
def test_binary_classification(
    forest_binary_classification_with_test,
    svc_binary_classification_with_test,
    pos_label,
    nb_stats,
    data_source,
):
    """Check the behaviour of the `MetricsSummaryDisplay` method with binary
    classification. We test both with an SVC that does not support `predict_proba` and a
    RandomForestClassifier that does.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    result = report.metrics.summarize(
        pos_label=pos_label, data_source=data_source, **kwargs
    ).frame()
    assert "Favorability" not in result.columns
    expected_metrics = (
        "precision",
        "recall",
        "roc_auc",
        "brier_score",
        "fit_time",
        "predict_time",
    )
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 4
    _check_results_summarize(result, expected_metrics, expected_nb_stats)

    # Repeat the same experiment where we the target labels are not [0, 1] but
    # ["neg", "pos"]. We check that we don't get any error.
    target_names = np.array(["neg", "pos"], dtype=object)
    pos_label_name = target_names[pos_label] if pos_label is not None else pos_label
    y_test = target_names[y_test]
    estimator = clone(estimator).fit(X_test, y_test)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    result = report.metrics.summarize(
        pos_label=pos_label_name, data_source=data_source, **kwargs
    ).frame()
    expected_metrics = (
        "precision",
        "recall",
        "roc_auc",
        "brier_score",
        "fit_time",
        "predict_time",
    )
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 4
    _check_results_summarize(result, expected_metrics, expected_nb_stats)

    estimator, X_test, y_test = svc_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    result = report.metrics.summarize(
        pos_label=pos_label, data_source=data_source, **kwargs
    ).frame()
    expected_metrics = ("precision", "recall", "roc_auc", "fit_time", "predict_time")
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 3
    _check_results_summarize(result, expected_metrics, expected_nb_stats)


@pytest.mark.parametrize("data_source", ["test", "X_y"])
def test_multiclass_classification(
    forest_multiclass_classification_with_test,
    svc_multiclass_classification_with_test,
    data_source,
):
    """Check the behaviour of the `MetricsSummaryDisplay` method with multiclass
    classification.
    """
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    result = report.metrics.summarize(data_source=data_source, **kwargs).frame()
    assert "Favorability" not in result.columns
    expected_metrics = (
        "precision",
        "recall",
        "roc_auc",
        "log_loss",
        "fit_time",
        "predict_time",
    )
    # since we are not averaging by default, we report 3 statistics for
    # precision, recall and roc_auc
    expected_nb_stats = 3 * 3 + 3
    _check_results_summarize(result, expected_metrics, expected_nb_stats)

    estimator, X_test, y_test = svc_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    result = report.metrics.summarize(data_source=data_source, **kwargs).frame()
    expected_metrics = ("precision", "recall", "fit_time", "predict_time")
    # since we are not averaging by default, we report 3 statistics for
    # precision and recall
    expected_nb_stats = 3 * 2 + 2
    _check_results_summarize(result, expected_metrics, expected_nb_stats)


@pytest.mark.parametrize("data_source", ["test", "X_y"])
def test_regression(linear_regression_with_test, data_source):
    """Check the behaviour of the `MetricsSummaryDisplay` method with regression."""
    estimator, X_test, y_test = linear_regression_with_test
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    result = report.metrics.summarize(data_source=data_source, **kwargs).frame()
    assert "Favorability" not in result.columns
    expected_metrics = ("r2", "rmse", "fit_time", "predict_time")
    _check_results_summarize(result, expected_metrics, len(expected_metrics))


def test_scoring_kwargs(
    linear_regression_multioutput_with_test, forest_multiclass_classification_with_test
):
    """Check the behaviour of the `MetricsSummaryDisplay` method with scoring kwargs."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, "summarize")
    result = report.metrics.summarize(
        metric_kwargs={"multioutput": "raw_values"}
    ).frame()
    assert result.shape == (6, 1)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Metric", "Output"]

    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, "summarize")
    result = report.metrics.summarize(metric_kwargs={"average": None}).frame()
    assert result.shape == (12, 1)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Metric", "Label / Average"]


@pytest.mark.parametrize(
    "fixture_name, scoring_dict, expected_columns",
    [
        (
            "linear_regression_with_test",
            {
                "R2": "r2",
                "RMSE": "rmse",
                "FIT_TIME": "_fit_time",
                "PREDICT_TIME": "_predict_time",
            },
            ["R2", "RMSE", "FIT_TIME", "PREDICT_TIME"],
        ),
        (
            "forest_multiclass_classification_with_test",
            {
                "Precision": "_precision",
                "Recall": "_recall",
                "ROC AUC": "_roc_auc",
                "Log Loss": "_log_loss",
                "Fit Time": "_fit_time",
                "Predict Time": "_predict_time",
            },
            [
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
def test_overwrite_scoring_names_with_dict(
    request, fixture_name, scoring_dict, expected_columns
):
    """Test that we can overwrite the scoring names using dict scoring."""
    estimator, X_test, y_test = request.getfixturevalue(fixture_name)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    result = report.metrics.summarize(metric=scoring_dict).frame()
    assert result.shape == (len(expected_columns), 1)

    # Get level 0 names if MultiIndex, otherwise get column names
    result_index = (
        result.index.get_level_values(0).tolist()
        if isinstance(result.index, pd.MultiIndex)
        else result.index.tolist()
    )
    assert result_index == expected_columns


def test_indicator_favorability(
    forest_binary_classification_with_test,
):
    """Check that the behaviour of `indicator_favorability` is correct."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    result = report.metrics.summarize(indicator_favorability=True).frame()
    assert "Favorability" in result.columns
    indicator = result["Favorability"]
    assert indicator["Precision"].tolist() == ["(↗︎)", "(↗︎)"]
    assert indicator["Recall"].tolist() == ["(↗︎)", "(↗︎)"]
    assert indicator["ROC AUC"].tolist() == ["(↗︎)"]
    assert indicator["Brier score"].tolist() == ["(↘︎)"]


@pytest.mark.parametrize(
    "scoring, scoring_kwargs",
    [
        ("accuracy", None),
        ("neg_log_loss", None),
        (accuracy_score, {"response_method": "predict"}),
        (get_scorer("accuracy"), None),
    ],
)
def test_scoring_single_list_equivalence(
    forest_binary_classification_with_test, scoring, scoring_kwargs
):
    """Check that passing a single string, callable, scorer is equivalent to passing a
    list with a single element."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    result_single = report.metrics.summarize(
        metric=scoring, metric_kwargs=scoring_kwargs
    ).frame()
    result_list = report.metrics.summarize(
        metric=[scoring], metric_kwargs=scoring_kwargs
    ).frame()
    assert result_single.equals(result_list)


def test_scoring_custom_metric(linear_regression_with_test):
    """Check that we can pass a custom metric with specific kwargs into
    `MetricsSummaryDisplay`."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    weights = np.ones_like(y_test) * 2

    def custom_metric(y_true, y_pred, some_weights):
        return np.mean((y_true - y_pred) * some_weights)

    result = report.metrics.summarize(
        metric=["r2", custom_metric],
        metric_kwargs={"some_weights": weights, "response_method": "predict"},
    ).frame()
    assert result.shape == (2, 1)
    np.testing.assert_allclose(
        result.to_numpy(),
        [
            [r2_score(y_test, estimator.predict(X_test))],
            [custom_metric(y_test, estimator.predict(X_test), weights)],
        ],
    )


def test_scorer(linear_regression_with_test):
    """Check that we can pass scikit-learn scorer with different parameters to
    the `MetricsSummaryDisplay` method."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    weights = np.ones_like(y_test) * 2

    def custom_metric(y_true, y_pred, some_weights):
        return np.mean((y_true - y_pred) * some_weights)

    median_absolute_error_scorer = make_scorer(
        median_absolute_error, response_method="predict"
    )
    custom_metric_scorer = make_scorer(
        custom_metric, response_method="predict", some_weights=weights
    )
    result = report.metrics.summarize(
        metric=[r2_score, median_absolute_error_scorer, custom_metric_scorer],
        metric_kwargs={"response_method": "predict"},  # only dispatched to r2_score
    ).frame()
    assert result.shape == (3, 1)
    np.testing.assert_allclose(
        result.to_numpy(),
        [
            [r2_score(y_test, estimator.predict(X_test))],
            [median_absolute_error(y_test, estimator.predict(X_test))],
            [custom_metric(y_test, estimator.predict(X_test), weights)],
        ],
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
    forest_binary_classification_with_test, scorer, pos_label
):
    """Check that we can pass scikit-learn scorer with different parameters to
    the `MetricsSummaryDisplay` method.

    We also check that we can pass `pos_label` whether to the scorer or to the
    `MetricsSummaryDisplay` method or consistently to both.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    result = report.metrics.summarize(
        metric=["accuracy", accuracy_score, scorer],
        metric_kwargs={"response_method": "predict"},
    ).frame()
    assert result.shape == (3, 1)
    np.testing.assert_allclose(
        result.to_numpy(),
        [
            [accuracy_score(y_test, estimator.predict(X_test))],
            [accuracy_score(y_test, estimator.predict(X_test))],
            [
                f1_score(
                    y_test,
                    estimator.predict(X_test),
                    average="macro",
                    pos_label=pos_label,
                )
            ],
        ],
    )


def test_scorer_pos_label_error(
    forest_binary_classification_with_test,
):
    """Check that we raise an error when pos_label is passed both in the scorer and
    globally conducting to a mismatch."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    f1_scorer = make_scorer(
        f1_score, response_method="predict", average="macro", pos_label=1
    )
    err_msg = re.escape(
        "`pos_label` is passed both in the scorer and to the `summarize` method."
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[f1_scorer], pos_label=0).frame()


def test_invalid_metric_type(linear_regression_with_test):
    """Check that we raise the expected error message if an invalid metric is passed."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    err_msg = re.escape("Invalid type of metric: <class 'int'> for 1")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[1]).frame()


def test_neg_metric_strings(
    forest_binary_classification_with_test,
):
    """Check that scikit-learn metrics with 'neg_' prefix are handled correctly."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    result = report.metrics.summarize(metric=["neg_log_loss"]).frame()
    assert "Log Loss" in result.index
    assert result.loc["Log Loss", "RandomForestClassifier"] == pytest.approx(
        report.metrics.log_loss()
    )


def test_sklearn_scoring_strings(
    forest_binary_classification_with_test,
):
    """Test that scikit-learn metric strings can be passed to summarize."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    class_report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    result = class_report.metrics.summarize(metric=["neg_log_loss"]).frame()
    assert "Log Loss" in result.index.get_level_values(0)

    result_multi = class_report.metrics.summarize(
        metric=["accuracy", "neg_log_loss", "roc_auc"], indicator_favorability=True
    ).frame()
    assert "Accuracy" in result_multi.index.get_level_values(0)
    assert "Log Loss" in result_multi.index.get_level_values(0)
    assert "ROC AUC" in result_multi.index.get_level_values(0)

    favorability = result_multi.loc["Accuracy"]["Favorability"]
    assert favorability == "(↗︎)"
    favorability = result_multi.loc["Log Loss"]["Favorability"]
    assert favorability == "(↘︎)"


def test_sklearn_scoring_strings_regression(
    linear_regression_with_test,
):
    """Test scikit-learn regression metric strings in `MetricsSummaryDisplay`."""
    regressor, X_test, y_test = linear_regression_with_test
    reg_report = EstimatorReport(regressor, X_test=X_test, y_test=y_test)

    reg_result = reg_report.metrics.summarize(
        metric=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
        indicator_favorability=True,
    ).frame()

    assert "Mean Squared Error" in reg_result.index.get_level_values(0)
    assert "Mean Absolute Error" in reg_result.index.get_level_values(0)
    assert "R²" in reg_result.index.get_level_values(0)

    assert reg_result.loc["Mean Squared Error"]["Favorability"] == "(↘︎)"
    assert reg_result.loc["R²"]["Favorability"] == "(↗︎)"


def test_scoring_strings_regression(linear_regression_with_test):
    """Test skore regression metric strings in `MetricsSummaryDisplay`."""
    regressor, X_test, y_test = linear_regression_with_test
    reg_report = EstimatorReport(regressor, X_test=X_test, y_test=y_test)

    reg_result = reg_report.metrics.summarize(
        metric=["rmse", "r2"], indicator_favorability=True
    ).frame()

    assert "RMSE" in reg_result.index.get_level_values(0)
    assert "R²" in reg_result.index.get_level_values(0)

    assert reg_result.loc["RMSE"]["Favorability"] == "(↘︎)"
    assert reg_result.loc["R²"]["Favorability"] == "(↗︎)"


def test_scorer_names_pos_label(
    forest_binary_classification_with_test,
):
    """Check that `pos_label` is dispatched with scikit-learn scorer names."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    result = report.metrics.summarize(metric=["f1"], pos_label=0).frame()
    assert "F1 Score" in result.index.get_level_values(0)
    assert 0 in result.index.get_level_values(1)
    f1_scorer = make_scorer(
        f1_score, response_method="predict", average="binary", pos_label=0
    )
    assert result.loc[("F1 Score", 0), "RandomForestClassifier"] == pytest.approx(
        f1_scorer(classifier, X_test, y_test)
    )


def test_sklearn_scorer_names_scoring_kwargs(
    forest_binary_classification_with_test,
):
    """Check that `metric_kwargs` is not supported when `metric` is a scikit-learn
    scorer name.
    """
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    err_msg = (
        "The `metric_kwargs` parameter is not supported when `metric` is a "
        "scikit-learn scorer name."
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(
            metric=["f1"], metric_kwargs={"average": "macro"}
        ).frame()


@pytest.mark.parametrize(
    "metric, metric_fn", [("precision", precision_score), ("recall", recall_score)]
)
def test_pos_label_overwrite(metric, metric_fn):
    """Check that `pos_label` can be overwritten in `summarize`"""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    classifier = LogisticRegression().fit(X, y)

    report = EstimatorReport(classifier, X_test=X, y_test=y)
    result = report.metrics.summarize(metric=metric).frame().reset_index()
    assert result["Label / Average"].to_list() == ["A", "B"]

    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="B")
    result = report.metrics.summarize(metric=metric).frame().reset_index()
    assert "Label / Average" not in result.columns
    assert result[report.estimator_name_].item() == pytest.approx(
        metric_fn(y, classifier.predict(X), pos_label="B")
    )

    result = (
        report.metrics.summarize(metric=metric, pos_label="A").frame().reset_index()
    )
    assert "Label / Average" not in result.columns
    assert result[report.estimator_name_].item() == pytest.approx(
        metric_fn(y, classifier.predict(X), pos_label="A")
    )
