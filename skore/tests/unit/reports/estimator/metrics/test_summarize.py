import re

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
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
    expected_estimator_name : str, optional
        Expected estimator name.
    expected_data_source : str, default="test"
        Expected data source value.
    """
    assert isinstance(display.data, pd.DataFrame)
    data = display.data

    assert set(data.columns) == {
        "metric",
        "estimator_name",
        "data_source",
        "label",
        "average",
        "output",
        "score",
        "favorability",
    }
    assert set(data["metric"]) == expected_metrics
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
            "Accuracy",
            "Precision",
            "Recall",
            "ROC AUC",
            "Log loss",
            "Brier score",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="RandomForestClassifier",
    )

    # Precision and recall should have 2 rows (one per class in binary classification)
    assert len(display.data[display.data["metric"] == "Precision"]) == 2
    assert len(display.data[display.data["metric"] == "Recall"]) == 2
    # Labels should be 0 and 1 for precision
    precision_labels = set(display.data[display.data["metric"] == "Precision"]["label"])
    assert precision_labels == {0, 1}

    # Output column should be all NaN for non-multioutput
    assert display.data["output"].isna().all()


def test_binary_classification_svc(svc_binary_classification_with_test):
    """
    Check the behaviour of summarize() with binary classification using SVC
    (no predict_proba).
    """
    estimator, X_test, y_test = svc_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test, pos_label=1)
    display = report.metrics.summarize()

    assert isinstance(display.data, pd.DataFrame)
    # No Brier score
    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "ROC AUC",
            "Fit time (s)",
            "Predict time (s)",
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
            "Accuracy",
            "Log loss",
            "Precision",
            "Recall",
            "ROC AUC",
            "Predict time (s)",
            "Fit time (s)",
        },
        expected_estimator_name="RandomForestClassifier",
    )

    assert display.data["output"].isna().all()
    data = display.data.set_index("metric")
    assert len(data.loc["Precision"]) == 3
    assert len(data.loc["Recall"]) == 3
    assert set(data.loc["Precision", "label"]) == {0, 1, 2}


def test_multiclass_classification_svc(svc_multiclass_classification_with_test):
    """Check the behaviour of summarize() with multiclass classification using SVC."""
    estimator, X_test, y_test = svc_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="SVC",
    )

    assert display.data["output"].isna().all()
    data = display.data.set_index("metric")
    assert len(data.loc["Precision"]) == 3
    assert len(data.loc["Recall"]) == 3
    assert set(data.loc["Precision", "label"]) == {0, 1, 2}


def test_regression(linear_regression_with_test):
    """Check the behaviour of summarize() with regression."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={"R²", "RMSE", "Fit time (s)", "Predict time (s)"},
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
        expected_metrics={"R²", "RMSE", "Fit time (s)", "Predict time (s)"},
        expected_estimator_name="LinearRegression",
    )

    assert display.data["label"].isna().all()
    data = display.data.set_index("metric")
    assert len(data.loc["R²", "output"]) == 2
    assert len(data.loc["RMSE", "output"]) == 2
    assert set(data.loc["R²", "output"]) == {0, 1}


def test_default_without_predict_proba(custom_classifier_no_predict_proba_with_test):
    """Default metrics skip roc_auc, log_loss, and brier_score without predict_proba."""
    estimator, X_test, y_test = custom_classifier_no_predict_proba_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="CustomClassifierPredictOnly",
    )


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
    assert_array_equal(train_data["score"], display_train.data["score"])

    test_data = display_both.data[display_both.data["data_source"] == "test"]
    assert_array_equal(test_data["score"], display_test.data["score"])


# Tests about passing `metric`


def test_invalid_metric_type(linear_regression_with_test):
    """Check that we raise the expected error message if an invalid metric is passed."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    err_msg = re.escape("Invalid type of metric: <class 'int'> for 1")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[1])


def test_empty_metric_list_uses_defaults(linear_regression_with_test):
    """Check that empty metric list falls back to default metrics."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    assert_frame_equal(
        report.metrics.summarize(metric=[]).data,
        report.metrics.summarize(metric=None).data,
    )


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
    assert set(display.data["metric"]) == {"RMSE", "R²"}


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
    assert set(display.data["metric"]) == set(metric_dict)


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
    assert set(display.data["metric"]) == set(metric_dict)


def test_metric_dict_overwrite_metric_names(forest_multiclass_classification_with_test):
    """Test that we can overwrite the metric names using dict metric."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    metric_dict = {"ROC AUC": "roc_auc", "Fit Time": "fit_time"}

    display = report.metrics.summarize(metric=metric_dict)
    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["metric"]) == set(metric_dict)


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
    assert set(display.data["metric"]) == {"R²", "Custom Metric"}

    scores = display.data.set_index("metric")["score"]
    assert scores["R²"] == pytest.approx(r2_score(y_test, estimator.predict(X_test)))
    assert scores["Custom Metric"] == pytest.approx(
        custom_metric(y_test, estimator.predict(X_test), weights)
    )


def test_custom_metric_average_none(forest_binary_classification_with_test):
    """
    Check that passing arguments to a custom metric through metric_kwargs
    works correctly.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def custom_f1(y_true, y_pred, average="binary"):
        return f1_score(y_true, y_pred, average=average)

    display = report.metrics.summarize(
        metric=custom_f1, response_method="predict", metric_kwargs={"average": None}
    )

    assert len(display.data) == 2
    assert display.data["average"].isna().all()
    assert set(display.data["label"]) == {0, 1}


def test_custom_metric_no_response_method(forest_binary_classification_with_test):
    """Test that ValueError is raised when callable metric lacks response_method."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def custom_metric(y_true, y_pred):
        return 0.5

    with pytest.raises(ValueError, match="response_method is required"):
        report.metrics.summarize(metric=custom_metric)


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
        (make_scorer(f1_score, response_method="predict", average="macro"), 1),
    ],
)
def test_scorer_binary_classification(
    forest_binary_classification_with_test, scorer, pos_label
):
    """Check that we can pass scikit-learn scorer with different parameters to
    summarize()."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(
        estimator, X_test=X_test, y_test=y_test, pos_label=pos_label
    )

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
            y_test,
            estimator.predict(X_test),
            average="macro",
        ),
    ]
    np.testing.assert_allclose(display.data["score"].values, expected_scores)


def test_scorer_with_average(
    forest_multiclass_classification_with_test,
):
    """Test multiclass classification with average parameter."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    scorer = make_scorer(f1_score, average="macro")
    display = report.metrics.summarize(metric=[scorer])

    assert len(display.data) == 1
    assert display.data["average"].values[0] == "macro"
    assert display.data["label"].isna().all()


def test_neg_metric_strings(forest_binary_classification_with_test):
    """Check that scikit-learn metrics with 'neg_' prefix are handled correctly."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=["neg_log_loss"])
    assert isinstance(display.data, pd.DataFrame)

    # Note: neg_log_loss was converted to log_loss
    assert "Log Loss" in set(display.data["metric"])

    score = display.data.set_index("metric").loc["Log Loss", "score"]
    assert score == pytest.approx(report.metrics.log_loss())


def test_sklearn_metric_strings(forest_binary_classification_with_test):
    """Check that multiple scikit-learn metric strings can be passed to summarize."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=["rand_score", "v_measure_score"])
    assert set(display.data["metric"]) == {"Rand Score", "V Measure Score"}


def test_sklearn_metric_strings_regression(linear_regression_with_test):
    """Test scikit-learn regression metric strings in summarize()."""
    regressor, X_test, y_test = linear_regression_with_test
    reg_report = EstimatorReport(regressor, X_test=X_test, y_test=y_test)

    display = reg_report.metrics.summarize(
        metric=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
    )

    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["metric"]) == {
        "Mean Squared Error",
        "Mean Absolute Error",
        "R²",
    }


def test_unknown_ml_task(forest_binary_classification_with_test):
    """Test summarize with unknown ML task."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    report._ml_task = "unknown-task"

    # If ML task is not recognized then none of the default metrics will
    # work
    def custom_metric(y_true, y_pred):
        return 0.8

    display = report.metrics.summarize(
        metric=[custom_metric], response_method="predict"
    )

    assert len(display.data) == 1
    assert display.data["score"].values[0] == 0.8
    assert display.data["label"].isna().all()
    assert display.data["average"].isna().all()
    assert display.data["output"].isna().all()


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
    assert len(display.data[display.data["metric"] == "R²"]) >= 2


def test_metric_kwargs_none(
    forest_binary_classification_with_test,
):
    """Test callable metric when metric_kwargs is None."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def custom_metric(y_true, y_pred):
        return 0.75

    display = report.metrics.summarize(metric=custom_metric, response_method="predict")

    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 1
    assert display.data["score"].values[0] == 0.75


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


def test_pos_label(forest_binary_classification_with_test):
    """Check that `pos_label` can be passed."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test, pos_label=1)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "ROC AUC",
            "Log loss",
            "Brier score",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="RandomForestClassifier",
    )

    assert len(display.data[display.data["metric"] == "Precision"]) == 1
    assert len(display.data[display.data["metric"] == "Recall"]) == 1
    # Label is None for precision/recall when pos_label is specified
    assert display.data["label"].isna().all()
    assert display.data["output"].isna().all()


def test_pos_label_scorer_error(forest_binary_classification_with_test):
    """Check that we raise an error when pos_label is passed both in the scorer and
    in summarize()."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test, pos_label=0)

    f1_scorer = make_scorer(
        f1_score, response_method="predict", average="macro", pos_label=1
    )
    err_msg = re.escape(
        "The `pos_label` passed in the scorer and the one used when creating the "
        "report must match; got 1 and 0."
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[f1_scorer])


def test_pos_label_strings(forest_binary_classification_with_test):
    """
    Check the behaviour of summarize() with binary classification using string labels.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test

    target_names = np.array(["neg", "pos"], dtype=object)
    y_test = target_names[y_test]

    estimator = clone(estimator).fit(X_test, y_test)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize()
    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["metric"]) == {
        "Accuracy",
        "Precision",
        "Recall",
        "ROC AUC",
        "Log loss",
        "Brier score",
        "Fit time (s)",
        "Predict time (s)",
    }

    labels = display.data.set_index("metric").loc["Precision", "label"]
    assert set(labels) == {"neg", "pos"}


def test_pos_label_bool(forest_binary_classification_with_test):
    """
    Check the behaviour of summarize() with binary classification using boolean labels.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test

    target_names = np.array([False, True], dtype=bool)
    y_test = target_names[y_test]

    estimator = clone(estimator).fit(X_test, y_test)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize()
    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["metric"]) == {
        "Accuracy",
        "Precision",
        "Recall",
        "ROC AUC",
        "Log loss",
        "Brier score",
        "Fit time (s)",
        "Predict time (s)",
    }

    labels = display.data.set_index("metric").loc["Precision", "label"]
    # Use `is` to avoid casting
    assert any(label is np.False_ for label in labels)
    assert any(label is np.True_ for label in labels)


def test_pos_label_scorer_names(
    forest_binary_classification_with_test,
):
    """Check that `pos_label` is dispatched with scikit-learn scorer names."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test, pos_label=0)

    display = report.metrics.summarize(metric=["f1"])
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
    """Check that `pos_label` can be set when creating the report."""
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

    # Test with pos_label="A" - should have single row
    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="A")
    display = report.metrics.summarize(metric=metric)
    assert len(display.data) == 1
    score_A = display.data["score"].values[0]
    assert score_A == pytest.approx(metric_fn(y, classifier.predict(X), pos_label="A"))


# Tests for issues #2203 and #2204: scorer API ergonomics


def test_scorer_response_method_not_required_in_summarize(linear_regression_with_test):
    """Regression test for #2203: when passing a scorer to summarize(), the
    response_method embedded in the scorer should be used automatically —
    the user should not need to pass it a second time."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def business_loss(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    scorer = make_scorer(
        business_loss, greater_is_better=False, response_method="predict"
    )

    # response_method is already in the scorer — no need to pass it explicitly
    display = report.metrics.summarize(metric=scorer)

    assert len(display.data) == 1
    expected = business_loss(y_test, estimator.predict(X_test))
    assert display.data["score"].iloc[0] == pytest.approx(expected)


def test_custom_metric_with_scorer_no_attribute_error(linear_regression_with_test):
    """Regression test for #2204: passing a make_scorer object to custom_metric()
    used to raise AttributeError because the code accessed ._score_func.__name__
    instead of scorer.__name__."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def business_loss(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    scorer = make_scorer(
        business_loss, greater_is_better=False, response_method="predict"
    )

    result = report.metrics.custom_metric(
        metric_function=scorer, response_method="predict"
    )
    assert result == pytest.approx(business_loss(y_test, estimator.predict(X_test)))


def test_scorer_metric_kwargs_override(linear_regression_with_test):
    """Regression test for #2203 follow-up: when a scorer has kwargs baked in,
    passing metric_kwargs to summarize() should override them."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    weights_in_scorer = np.ones_like(y_test)
    weights_override = np.ones_like(y_test) * 2

    def weighted_mae(y_true, y_pred, sample_weight=None):
        return np.average(np.abs(y_true - y_pred), weights=sample_weight)

    scorer = make_scorer(
        weighted_mae, response_method="predict", sample_weight=weights_in_scorer
    )

    display = report.metrics.summarize(
        metric=scorer, metric_kwargs={"sample_weight": weights_override}
    )

    expected = weighted_mae(
        y_test, estimator.predict(X_test), sample_weight=weights_override
    )
    assert display.data["score"].iloc[0] == pytest.approx(expected)
