import re

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
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
from sklearn.model_selection import train_test_split

from skore import EstimatorReport, MetricsSummaryDisplay
from skore._utils._testing import check_cache_changed, check_cache_unchanged


def check_display_structure(
    display,
    *,
    expected_metrics,
    expected_verbose_names=None,
    expected_estimator_name=None,
    expected_data_source="test",
):
    """
    Helper function to check the structure of a MetricsSummaryDisplay.data DataFrame.

    Parameters
    ----------
    display : MetricsSummaryDisplay
        The display object to check.
    expected_metrics : set, optional
        Expected set of metric names.
    expected_verbose_names : set, optional
        Expected set of verbose names.
    expected_estimator_name : str, optional
        Expected estimator name.
    expected_data_source : str, default="test"
        Expected data source value.
    """
    assert isinstance(display.data, pd.DataFrame)
    data = display.data

    assert set(data.columns) == {
        "metric",
        "verbose_name",
        "estimator_name",
        "data_source",
        "label",
        "average",
        "output",
        "score",
        "favorability",
    }
    assert set(data["metric"]) == expected_metrics
    if expected_verbose_names is not None:
        assert set(data["verbose_name"]) == expected_verbose_names
    assert set(data["estimator_name"]) == {expected_estimator_name}
    assert set(data["data_source"]) == {expected_data_source}
    assert data["average"].isna().all()
    assert pd.api.types.is_numeric_dtype(data["score"])
    assert set(data["favorability"]) == {"(↗︎)", "(↘︎)"}


# Tests for the happy path, with different ML tasks


def test_binary_classification_forest(forest_binary_classification_with_test):
    """
    Check the behaviour of summarize() with binary classification using
    RandomForestClassifier.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "accuracy",
            "precision",
            "recall",
            "roc_auc",
            "brier_score",
            "fit_time",
            "predict_time",
        },
        expected_estimator_name="RandomForestClassifier",
    )

    # Precision and recall should have 2 rows (one per class in binary classification)
    assert len(display.data[display.data["metric"] == "precision"]) == 2
    assert len(display.data[display.data["metric"] == "recall"]) == 2
    # Labels should be 0 and 1 for precision
    precision_labels = set(display.data[display.data["metric"] == "precision"]["label"])
    assert precision_labels == {0, 1}

    # Output column should be all NaN for non-multioutput
    assert display.data["output"].isna().all()


def test_binary_classification_forest_pos_label(forest_binary_classification_with_test):
    """
    Check the behaviour of summarize() with binary classification using
    RandomForestClassifier with pos_label specified.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize(pos_label=1)

    check_display_structure(
        display,
        expected_metrics={
            "accuracy",
            "precision",
            "recall",
            "roc_auc",
            "brier_score",
            "fit_time",
            "predict_time",
        },
        expected_estimator_name="RandomForestClassifier",
    )

    assert len(display.data[display.data["metric"] == "precision"]) == 1
    assert len(display.data[display.data["metric"] == "recall"]) == 1
    # Label is None for precision/recall when pos_label is specified
    assert display.data["label"].isna().all()
    assert display.data["output"].isna().all()


def test_binary_classification_svc(svc_binary_classification_with_test):
    """
    Check the behaviour of summarize() with binary classification using SVC
    (no predict_proba).
    """
    estimator, X_test, y_test = svc_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize(pos_label=1)

    assert isinstance(display.data, pd.DataFrame)
    # No Brier score
    check_display_structure(
        display,
        expected_metrics={
            "accuracy",
            "precision",
            "recall",
            "roc_auc",
            "fit_time",
            "predict_time",
        },
        expected_estimator_name="SVC",
    )


def test_multiclass_classification_forest(forest_multiclass_classification_with_test):
    """
    Check the behaviour of summarize() with multiclass classification using
    RandomForestClassifier.
    """
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "accuracy",
            "log_loss",
            "precision",
            "recall",
            "roc_auc",
            "predict_time",
            "fit_time",
        },
        expected_estimator_name="RandomForestClassifier",
    )

    assert display.data["output"].isna().all()
    data = display.data.set_index("metric")
    assert len(data.loc["precision"]) == 3
    assert len(data.loc["recall"]) == 3
    assert set(data.loc["precision", "label"]) == {0, 1, 2}


def test_multiclass_classification_svc(svc_multiclass_classification_with_test):
    """Check the behaviour of summarize() with multiclass classification using SVC."""
    estimator, X_test, y_test = svc_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "accuracy",
            "precision",
            "recall",
            "fit_time",
            "predict_time",
        },
        expected_estimator_name="SVC",
    )

    assert display.data["output"].isna().all()
    data = display.data.set_index("metric")
    assert len(data.loc["precision"]) == 3
    assert len(data.loc["recall"]) == 3
    assert set(data.loc["precision", "label"]) == {0, 1, 2}


def test_regression(linear_regression_with_test):
    """Check the behaviour of summarize() with regression."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={"r2", "rmse", "fit_time", "predict_time"},
        expected_verbose_names={"R²", "RMSE", "Fit time (s)", "Predict time (s)"},
        expected_estimator_name="LinearRegression",
    )

    assert display.data["label"].isna().all()
    assert display.data["output"].isna().all()


def test_multioutput_regression(linear_regression_multioutput_with_test):
    """Check the behaviour of summarize() with multioutput regression."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize(metric_kwargs={"multioutput": "raw_values"})

    check_display_structure(
        display,
        expected_metrics={"r2", "rmse", "fit_time", "predict_time"},
        expected_verbose_names={"R²", "RMSE", "Fit time (s)", "Predict time (s)"},
        expected_estimator_name="LinearRegression",
    )

    assert display.data["label"].isna().all()
    data = display.data.set_index("metric")
    assert len(data.loc["r2", "output"]) == 2
    assert len(data.loc["rmse", "output"]) == 2
    assert set(data.loc["r2", "output"]) == {0, 1}


def test_cache(forest_binary_classification_with_test):
    """Check the behaviour of the metrics methods available for binary
    classification.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    with check_cache_changed(report._cache):
        result = report.metrics.summarize()
    assert isinstance(result, MetricsSummaryDisplay)

    with check_cache_unchanged(report._cache):
        result_from_cache = report.metrics.summarize()
    assert_frame_equal(result.data, result_from_cache.data)

    report.clear_cache()

    # Passing data from outside touches the cache
    with check_cache_changed(report._cache):
        result_external_data = report.metrics.summarize(
            data_source="X_y",
            X=X_test,
            y=y_test,
        )
    assert isinstance(result_external_data, MetricsSummaryDisplay)

    # predict_time is not deterministic
    assert_frame_equal(
        result.data[result.data["metric"] != "predict_time"].drop(
            columns="data_source"
        ),
        result_external_data.data[
            result_external_data.data["metric"] != "predict_time"
        ].drop(columns="data_source"),
    )


def test_data_source_both(forest_binary_classification_data):
    """Check the behaviour with `data_source="both"`."""
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display_train = report.metrics.summarize(data_source="train")
    display_test = report.metrics.summarize(data_source="test")
    display_both = report.metrics.summarize(data_source="both")

    assert set(display_both.data["data_source"]) == {"train", "test"}

    train_data = display_both.data[display_both.data["data_source"] == "train"]
    assert_series_equal(
        train_data["score"], display_train.data["score"], check_index=False
    )

    test_data = display_both.data[display_both.data["data_source"] == "test"]
    assert_series_equal(
        test_data["score"], display_test.data["score"], check_index=False
    )


# Tests about passing `metric`


def test_invalid_metric_type(linear_regression_with_test):
    """Check that we raise the expected error message if an invalid metric is passed."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    err_msg = re.escape("Invalid type of metric: <class 'int'> for 1")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[1])


@pytest.mark.parametrize("metric", ["public_metric", "_private_metric"])
def test_error_metric_strings(linear_regression_with_test, metric):
    """Check that we raise an error if a metric string is not a valid metric."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    err_msg = re.escape(f"Invalid metric: {metric!r}.")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[metric])


def test_metric_strings_regression(linear_regression_with_test):
    """Test skore regression metric strings in summarize()."""
    regressor, X_test, y_test = linear_regression_with_test
    reg_report = EstimatorReport(regressor, X_test=X_test, y_test=y_test)

    display = reg_report.metrics.summarize(metric=["rmse", "r2"])

    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["verbose_name"]) == {"RMSE", "R²"}


def test_metric_dict(forest_binary_classification_with_test):
    """Test that metric can be passed as a dictionary with custom names."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    metric_dict = {
        "Custom Accuracy": "accuracy",
        "Custom Precision": "precision",
        "Custom R2": get_scorer("neg_mean_absolute_error"),
    }

    display = report.metrics.summarize(metric=metric_dict)
    assert isinstance(display.data, pd.DataFrame)

    # Precision will have 2 rows (one per class)
    assert len(display.data) == 4
    assert set(display.data["verbose_name"]) == set(metric_dict)


def test_metric_dict_with_callables(linear_regression_with_test):
    """Test that metric dict works with callable functions."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def custom_metric(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    metric_dict = {"R Squared": "r2", "Custom MAE": custom_metric}

    display = report.metrics.summarize(
        metric=metric_dict, metric_kwargs={"response_method": "predict"}
    )
    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 2
    assert set(display.data["verbose_name"]) == set(metric_dict)


def test_metric_dict_overwrite_metric_names(forest_multiclass_classification_with_test):
    """Test that we can overwrite the metric names using dict metric."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    metric_dict = {"ROC AUC": "_roc_auc", "Fit Time": "_fit_time"}

    display = report.metrics.summarize(metric=metric_dict)
    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["verbose_name"]) == set(metric_dict)


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
    forest_binary_classification_with_test, metric, metric_kwargs
):
    """Check that passing a single string, callable, scorer is equivalent to passing a
    list with a single element."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display_single = report.metrics.summarize(
        metric=metric, metric_kwargs=metric_kwargs
    )
    display_list = report.metrics.summarize(
        metric=[metric], metric_kwargs=metric_kwargs
    )
    pd.testing.assert_frame_equal(display_single.data, display_list.data)


def test_custom_metric(linear_regression_with_test):
    """Check that we can pass a custom metric with specific kwargs to summarize()."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    weights = np.ones_like(y_test) * 2

    def custom_metric(y_true, y_pred, some_weights):
        return np.mean((y_true - y_pred) * some_weights)

    display = report.metrics.summarize(
        metric=["r2", custom_metric],
        metric_kwargs={"some_weights": weights, "response_method": "predict"},
    )
    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["metric"]) == {"r2", "custom_metric"}

    scores = display.data.set_index("metric")["score"]
    assert scores["r2"] == pytest.approx(r2_score(y_test, estimator.predict(X_test)))
    assert scores["custom_metric"] == pytest.approx(
        custom_metric(y_test, estimator.predict(X_test), weights)
    )


def test_scorer(linear_regression_with_test):
    """
    Check that we can pass scikit-learn scorer with different parameters to summarize().
    """
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    weights = np.ones_like(y_test) * 2

    def custom_metric(y_true, y_pred, some_weights):
        return np.mean((y_true - y_pred) * some_weights)

    custom_metric_scorer = make_scorer(
        custom_metric, response_method="predict", some_weights=weights
    )

    median_absolute_error_scorer = make_scorer(
        median_absolute_error, response_method="predict"
    )

    display = report.metrics.summarize(
        metric=[r2_score, median_absolute_error_scorer, custom_metric_scorer],
        metric_kwargs={"response_method": "predict"},
    )
    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 3

    scores = display.data["score"].values
    expected_scores = [
        r2_score(y_test, estimator.predict(X_test)),
        median_absolute_error(y_test, estimator.predict(X_test)),
        custom_metric(y_test, estimator.predict(X_test), weights),
    ]
    np.testing.assert_allclose(scores, expected_scores)


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
    summarize()."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(
        metric=["accuracy", accuracy_score, scorer],
        metric_kwargs={"response_method": "predict"},
    )
    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 3

    expected_scores = [
        accuracy_score(y_test, estimator.predict(X_test)),
        accuracy_score(y_test, estimator.predict(X_test)),
        f1_score(
            y_test, estimator.predict(X_test), average="macro", pos_label=pos_label
        ),
    ]
    np.testing.assert_allclose(display.data["score"].values, expected_scores)


def test_neg_metric_strings(forest_binary_classification_with_test):
    """Check that scikit-learn metrics with 'neg_' prefix are handled correctly."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=["neg_log_loss"])
    assert isinstance(display.data, pd.DataFrame)

    # Note: neg_log_loss was converted to log_loss
    assert "log_loss" in set(display.data["metric"])
    assert "Log Loss" in set(display.data["verbose_name"])

    score = display.data.set_index("metric").loc["log_loss", "score"]
    assert score == pytest.approx(report.metrics.log_loss())


def test_sklearn_metric_strings(forest_binary_classification_with_test):
    """Check that multiple scikit-learn metric strings can be passed to summarize."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=["accuracy", "log_loss", "roc_auc"])
    assert set(display.data["metric"]) == {"accuracy", "log_loss", "roc_auc"}
    assert set(display.data["verbose_name"]) == {"Accuracy", "Log loss", "ROC AUC"}


def test_sklearn_metric_strings_regression(
    linear_regression_with_test,
):
    """Test scikit-learn regression metric strings in summarize()."""
    regressor, X_test, y_test = linear_regression_with_test
    reg_report = EstimatorReport(regressor, X_test=X_test, y_test=y_test)

    display = reg_report.metrics.summarize(
        metric=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
    )

    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["verbose_name"]) == {
        "Mean Squared Error",
        "Mean Absolute Error",
        "R²",
    }


# Tests about passing `metric_kwargs`


def test_metric_kwargs_average(forest_multiclass_classification_with_test):
    """Check the behaviour of summarize() when `average` is passed in metric_kwargs."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize(metric_kwargs={"average": None})

    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) > 4


def test_metric_kwargs_multioutput(linear_regression_multioutput_with_test):
    """Check the behaviour of summarize() when `multioutput` is passed in
    metric_kwargs."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, "summarize")
    display = report.metrics.summarize(metric_kwargs={"multioutput": "raw_values"})

    assert isinstance(display.data, pd.DataFrame)
    assert "output" in display.data.columns
    assert len(display.data[display.data["metric"] == "r2"]) >= 2


def test_sklearn_scorer_names_metric_kwargs(forest_binary_classification_with_test):
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
        report.metrics.summarize(metric=["f1"], metric_kwargs={"average": "macro"})


# Tests about passing `pos_label`


def test_pos_label_scorer_error(forest_binary_classification_with_test):
    """Check that we raise an error when pos_label is passed both in the scorer and
    in summarize()."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    f1_scorer = make_scorer(
        f1_score, response_method="predict", average="macro", pos_label=1
    )
    err_msg = re.escape(
        "`pos_label` is passed both in the scorer and to the `summarize` method."
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[f1_scorer], pos_label=0)


@pytest.mark.parametrize("pos_label", [None, 1])
def test_pos_label_strings(forest_binary_classification_with_test, pos_label):
    """
    Check the behaviour of summarize() with binary classification using string labels.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test

    target_names = np.array(["neg", "pos"], dtype=object)
    pos_label_string = target_names[pos_label] if pos_label is not None else pos_label
    y_test = target_names[y_test]

    estimator = clone(estimator).fit(X_test, y_test)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(pos_label=pos_label_string)
    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["metric"]) == {
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "brier_score",
        "fit_time",
        "predict_time",
    }


def test_pos_label_scorer_names(
    forest_binary_classification_with_test,
):
    """Check that `pos_label` is dispatched with scikit-learn scorer names."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=["f1"], pos_label=0)
    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["label"]) == {0}

    f1_scorer = make_scorer(
        f1_score, response_method="predict", average="binary", pos_label=0
    )
    score = display.data["score"].to_list()[0]
    assert score == pytest.approx(f1_scorer(classifier, X_test, y_test))


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

    # Test without pos_label - should have multiple rows (one per class)
    report = EstimatorReport(classifier, X_test=X, y_test=y)
    display = report.metrics.summarize(metric=metric)
    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 2  # One row per class
    assert set(display.data["label"]) == {"A", "B"}

    # Test with pos_label="B" - should have single row
    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="B")
    display = report.metrics.summarize(metric=metric)
    assert len(display.data) == 1
    score_B = display.data["score"].values[0]
    assert score_B == pytest.approx(metric_fn(y, classifier.predict(X), pos_label="B"))

    # Test with pos_label="A" override - should have single row
    display = report.metrics.summarize(metric=metric, pos_label="A")
    assert len(display.data) == 1
    score_A = display.data["score"].values[0]
    assert score_A == pytest.approx(metric_fn(y, classifier.predict(X), pos_label="A"))
