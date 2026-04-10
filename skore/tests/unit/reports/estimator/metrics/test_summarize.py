"""Tests for EstimatorReport.metrics.summarize().

Organised by metric input type, then corner cases:

- Default metrics — by ML task variant
- Metric strings — skore built-ins and sklearn scorer names
- Metric scorers — make_scorer / get_scorer instances
- Metric callables — plain functions
- Mixed metric types — strings + scorers + callables in one call
- metric_kwargs
- Cache and data_source
"""

import re

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    make_scorer,
    mean_absolute_error,
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
    expected_favorability=None,
    expected_average=None,
):
    """Check the full structure of a MetricsSummaryDisplay.data DataFrame."""
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
    assert pd.api.types.is_numeric_dtype(data["score"])
    assert set(data["metric"]) == expected_metrics
    assert set(data["estimator_name"]) == {expected_estimator_name}
    assert set(data["data_source"]) == {expected_data_source}
    if expected_average is None:
        assert data["average"].isna().all()
    else:
        assert set(data["average"]) == expected_average
    if expected_favorability is None:
        expected_favorability = {"(↗︎)", "(↘︎)"}
    assert set(data["favorability"]) == expected_favorability


# Default metrics


@pytest.mark.parametrize("metric", [None, [], {}])
def test_default_plain(forest_binary_classification_with_test, metric):
    """metric=None / [] / {} selects the canonical defaults for the ML task."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=metric)

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


def test_default_with_extra_args(linear_regression_multioutput_with_test):
    """metric=None with metric_kwargs alters how the default metrics are computed.

    For multioutput regression, multioutput='raw_values' produces one row per
    output for R² and RMSE.
    """
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric_kwargs={"multioutput": "raw_values"})

    assert isinstance(display, MetricsSummaryDisplay)
    r2_rows = display.data[display.data["metric"] == "R²"]
    rmse_rows = display.data[display.data["metric"] == "RMSE"]
    assert len(r2_rows) >= 2
    assert len(rmse_rows) >= 2


def test_default_as_dict(linear_regression_with_test):
    """Built-in metric names as a dict: dict keys become the display names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric={"My R²": "r2", "My RMSE": "rmse"})

    check_display_structure(
        display,
        expected_metrics={"My R²", "My RMSE"},
        expected_estimator_name="LinearRegression",
    )


def test_default_binary_classification_svc(svc_binary_classification_with_test):
    """SVC (no predict_proba): no ROC AUC, Log loss, or Brier score."""
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


def test_default_multiclass_classification_forest(
    forest_multiclass_classification_with_test,
):
    """Multiclass classification with RandomForestClassifier."""
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


def test_default_multiclass_classification_svc(svc_multiclass_classification_with_test):
    """Multiclass classification with SVC (no predict_proba)."""
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


def test_default_regression(linear_regression_with_test):
    """Regression with LinearRegression."""
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


def test_default_multioutput_regression(linear_regression_multioutput_with_test):
    """Multioutput regression with LinearRegression."""
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


def test_unknown_ml_task(forest_binary_classification_with_test):
    """Unknown ML task falls back to custom metric only."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    report._ml_task = "unknown-task"

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


def test_invalid_metric_type(linear_regression_with_test):
    """An integer in the metric list raises a clear ValueError."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    err_msg = re.escape("Invalid type of metric: <class 'int'> for 1")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[1])


# Metric string


def test_string_plain(linear_regression_with_test):
    """A list of skore built-in metric strings resolves to correct display names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=["r2", "rmse"])

    check_display_structure(
        display,
        expected_metrics={"R²", "RMSE"},
        expected_estimator_name="LinearRegression",
    )


def test_string_with_extra_args(forest_binary_classification_with_test):
    """Built-in metric strings with metric_kwargs change how the metric is computed.

    Passing average='macro' collapses per-class precision/recall to a single row.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(
        metric=["precision", "recall"],
        metric_kwargs={"average": "macro"},
    )

    check_display_structure(
        display,
        expected_metrics={"Precision", "Recall"},
        expected_estimator_name="RandomForestClassifier",
        expected_favorability={"(↗︎)"},
        expected_average={"macro"},
    )


def test_string_as_dict(linear_regression_with_test):
    """Metric strings in dict form: dict keys override display names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(
        metric={
            "Custom MAE": "neg_mean_absolute_error",
            "Custom MSE": "neg_mean_squared_error",
        }
    )

    check_display_structure(
        display,
        expected_metrics={"Custom MAE", "Custom MSE"},
        expected_estimator_name="LinearRegression",
        expected_favorability={"(↘︎)"},
    )


def test_sklearn_string(forest_binary_classification_with_test):
    """Multiple scikit-learn metric strings can be passed to summarize."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=["rand_score", "v_measure_score"])
    assert set(display.data["metric"]) == {"Rand Score", "V Measure Score"}


def test_sklearn_string_regression(linear_regression_with_test):
    """Scikit-learn regression metric strings in summarize()."""
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


def test_sklearn_string_neg(forest_binary_classification_with_test):
    """Scikit-learn metrics with 'neg_' prefix are handled correctly."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=["neg_log_loss"])
    assert isinstance(display.data, pd.DataFrame)

    assert "Log Loss" in set(display.data["metric"])

    score = display.data.set_index("metric").loc["Log Loss", "score"]
    assert score == pytest.approx(report.metrics.log_loss())


@pytest.mark.parametrize("metric", ["public_metric", "_private_metric"])
def test_string_unknown(linear_regression_with_test, metric):
    """An unrecognised metric string raises a clear ValueError."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    err_msg = re.escape(f"Invalid metric: {metric!r}.")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[metric])


def test_sklearn_string_metric_kwargs(forest_binary_classification_with_test):
    """metric_kwargs is not supported when metric is a sklearn scorer name string."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    err_msg = (
        "The `metric_kwargs` parameter is not supported when `metric` is a "
        "scikit-learn scorer name."
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=["f1"], metric_kwargs={"average": "macro"})


# Metric scorers


def test_scorer_plain(linear_regression_with_test):
    """A list of sklearn scorers produces the correct metric names and scores."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    r2_scorer = make_scorer(r2_score, response_method="predict")
    mae_scorer = make_scorer(
        mean_absolute_error, response_method="predict", greater_is_better=False
    )

    display = report.metrics.summarize(metric=[r2_scorer, mae_scorer])

    check_display_structure(
        display,
        expected_metrics={"R2 Score", "Mean Absolute Error"},
        expected_estimator_name="LinearRegression",
    )
    scores = display.data.set_index("metric")["score"]
    assert scores["R2 Score"] == pytest.approx(
        r2_score(y_test, estimator.predict(X_test))
    )
    assert scores["Mean Absolute Error"] == pytest.approx(
        mean_absolute_error(y_test, estimator.predict(X_test))
    )


def test_scorer_with_extra_args(forest_binary_classification_with_test):
    """Scorers created with make_scorer embed their own extra kwargs (e.g. average)."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    f1_scorer = make_scorer(f1_score, response_method="predict", average="macro")
    precision_scorer = make_scorer(
        precision_score,
        response_method="predict",
        average="weighted",
        zero_division=0,
    )

    display = report.metrics.summarize(metric=[f1_scorer, precision_scorer])

    check_display_structure(
        display,
        expected_metrics={"F1 Score", "Precision Score"},
        expected_estimator_name="RandomForestClassifier",
        expected_favorability={"(↗︎)"},
        expected_average={"macro", "weighted"},
    )
    averages = display.data.set_index("metric")["average"]
    assert averages["F1 Score"] == "macro"
    assert averages["Precision Score"] == "weighted"


def test_scorer_as_dict(linear_regression_with_test):
    """Scorers in dict form: dict keys become the display names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    r2_scorer = make_scorer(r2_score, response_method="predict")
    mae_scorer = make_scorer(
        mean_absolute_error, response_method="predict", greater_is_better=False
    )

    display = report.metrics.summarize(
        metric={"Custom R2": r2_scorer, "Custom MAE": mae_scorer}
    )

    check_display_structure(
        display,
        expected_metrics={"Custom R2", "Custom MAE"},
        expected_estimator_name="LinearRegression",
        expected_favorability={"(↗︎)", "(↘︎)"},
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
        (make_scorer(f1_score, response_method="predict", average="macro"), 1),
    ],
)
def test_scorer_binary_classification(
    forest_binary_classification_with_test, scorer, pos_label
):
    """Scorers with different pos_label configurations in binary classification."""
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


def test_scorer_with_average(forest_multiclass_classification_with_test):
    """Multiclass classification with average parameter."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    scorer = make_scorer(f1_score, average="macro")
    display = report.metrics.summarize(metric=[scorer])

    assert len(display.data) == 1
    assert display.data["average"].values[0] == "macro"
    assert display.data["label"].isna().all()


def test_scorer_response_method_not_required_in_summarize(linear_regression_with_test):
    """response_method embedded in the scorer is used automatically.

    Regression test for #2203.
    """
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def business_loss(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    scorer = make_scorer(
        business_loss, greater_is_better=False, response_method="predict"
    )

    display = report.metrics.summarize(metric=scorer)

    assert len(display.data) == 1
    expected = business_loss(y_test, estimator.predict(X_test))
    assert display.data["score"].iloc[0] == pytest.approx(expected)


def test_scorer_pos_label_error(forest_binary_classification_with_test):
    """pos_label specified both in the scorer and in the report raises ValueError."""
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


# Metric callables


def test_callable_plain(linear_regression_with_test):
    """A list of plain callables requires response_method in metric_kwargs."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def my_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    display = report.metrics.summarize(
        metric=my_mae, metric_kwargs={"response_method": "predict"}
    )

    check_display_structure(
        display,
        expected_metrics={"My Mae"},
        expected_estimator_name="LinearRegression",
        expected_favorability={""},
    )
    assert set(display.data["favorability"]) == {""}
    score = display.data["score"].values[0]
    assert score == pytest.approx(
        mean_absolute_error(y_test, estimator.predict(X_test))
    )


def test_callable_with_extra_args(linear_regression_with_test):
    """
    Callables that need extra keyword arguments receive them through metric_kwargs.
    """
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    weights = np.ones_like(y_test) * 3.0

    def weighted_mae(y_true, y_pred, weights):
        return float(np.average(np.abs(y_true - y_pred), weights=weights))

    display = report.metrics.summarize(
        metric=weighted_mae,
        metric_kwargs={"response_method": "predict", "weights": weights},
    )

    check_display_structure(
        display,
        expected_metrics={"Weighted Mae"},
        expected_estimator_name="LinearRegression",
        expected_favorability={""},
    )
    score = display.data["score"].values[0]
    assert score == pytest.approx(
        weighted_mae(y_test, estimator.predict(X_test), weights)
    )


def test_callable_as_dict(linear_regression_with_test):
    """Callables in dict form: dict keys become the display names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def my_metric(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    display = report.metrics.summarize(
        metric={"My Custom Error": my_metric},
        metric_kwargs={"response_method": "predict"},
    )

    check_display_structure(
        display,
        expected_metrics={"My Custom Error"},
        expected_estimator_name="LinearRegression",
        expected_favorability={""},
    )
    score = display.data["score"].values[0]
    assert score == pytest.approx(
        mean_absolute_error(y_test, estimator.predict(X_test))
    )


def test_callable_no_response_method(forest_binary_classification_with_test):
    """A callable without response_method raises a descriptive ValueError."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def custom_metric(y_true, y_pred):
        return 0.5

    with pytest.raises(ValueError, match="response_method is required"):
        report.metrics.summarize(metric=custom_metric)


def test_callable_average_none(forest_binary_classification_with_test):
    """Passing arguments to a custom metric through metric_kwargs works correctly."""
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


# Mixed metric types


@pytest.mark.parametrize(
    "metric, metric_kwargs",
    [
        ("accuracy", None),
        ("neg_log_loss", None),
        (accuracy_score, {"response_method": "predict"}),
        (get_scorer("accuracy"), None),
    ],
)
def test_single_list_equivalence(
    forest_binary_classification_with_test, metric, metric_kwargs
):
    """Passing a single metric is equivalent to passing a list with one element."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display_single = report.metrics.summarize(
        metric=metric, metric_kwargs=metric_kwargs
    )
    display_list = report.metrics.summarize(
        metric=[metric], metric_kwargs=metric_kwargs
    )
    pd.testing.assert_frame_equal(display_single.data, display_list.data)


def test_mix_plain(linear_regression_with_test):
    """A list that mixes built-in string, scorer, and callable in one call."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    mae_scorer = make_scorer(
        mean_absolute_error, response_method="predict", greater_is_better=False
    )

    def my_r2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    display = report.metrics.summarize(
        metric=["rmse", mae_scorer, my_r2],
        metric_kwargs={"response_method": "predict"},
    )

    check_display_structure(
        display,
        expected_metrics={"RMSE", "Mean Absolute Error", "My R2"},
        expected_estimator_name="LinearRegression",
        expected_favorability={"", "(↘︎)"},
    )


def test_mix_with_extra_args(linear_regression_with_test):
    """A mixed list where the callable needs extra kwargs from metric_kwargs."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    mae_scorer = make_scorer(
        mean_absolute_error, response_method="predict", greater_is_better=False
    )

    def scaled_mae(y_true, y_pred, scale=1.0):
        return scale * mean_absolute_error(y_true, y_pred)

    display = report.metrics.summarize(
        metric=["r2", mae_scorer, scaled_mae],
        metric_kwargs={"response_method": "predict", "scale": 2.0},
    )

    check_display_structure(
        display,
        expected_metrics={"R²", "Mean Absolute Error", "Scaled Mae"},
        expected_estimator_name="LinearRegression",
        expected_favorability={"(↗︎)", "(↘︎)", ""},
    )
    scores = display.data.set_index("metric")["score"]
    plain_mae = mean_absolute_error(y_test, estimator.predict(X_test))
    assert scores["Scaled Mae"] == pytest.approx(2.0 * plain_mae)


def test_mix_as_dict(linear_regression_with_test):
    """A dict mixing built-in string, scorer, and callable with custom display names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    r2_scorer = make_scorer(r2_score, response_method="predict")

    def my_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    display = report.metrics.summarize(
        metric={
            "Skore RMSE": "rmse",
            "Scorer R2": r2_scorer,
            "Callable MAE": my_mae,
        },
        metric_kwargs={"response_method": "predict"},
    )

    check_display_structure(
        display,
        expected_metrics={"Skore RMSE", "Scorer R2", "Callable MAE"},
        expected_estimator_name="LinearRegression",
        expected_favorability={"(↗︎)", "(↘︎)", ""},
    )


# metric_kwargs


def test_metric_kwargs_average(forest_multiclass_classification_with_test):
    """average passed in metric_kwargs expands per-class metrics."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize(metric_kwargs={"average": None})

    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) > 4


def test_metric_kwargs_multioutput(linear_regression_multioutput_with_test):
    """multioutput passed in metric_kwargs produces per-output rows."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize(metric_kwargs={"multioutput": "raw_values"})

    assert isinstance(display.data, pd.DataFrame)
    assert "output" in display.data.columns
    assert len(display.data[display.data["metric"] == "R²"]) >= 2


def test_metric_kwargs_forwarded_to_score_func(
    forest_multiclass_classification_with_test,
):
    """metric_kwargs not in the subclass __call__ signature reach score_func.

    Non-regression test: zero_division is not in Precision.__call__'s signature,
    so it must be forwarded via **kwargs to actually affect the result.
    """

    # Estimator only predicts class 0
    _, X, y = forest_multiclass_classification_with_test
    estimator = DummyClassifier(strategy="constant", constant=0).fit(X, y)
    report = EstimatorReport(estimator, X_test=X, y_test=y)

    zero_division = 1
    display = report.metrics.summarize(
        metric="precision", metric_kwargs={"zero_division": zero_division}
    )

    # Estimator never predicts 1, so its precision on class 1 is equal to
    # zero_division
    class_1_score = display.data.query("label == 1")["score"].item()
    assert class_1_score == zero_division


def test_metric_kwargs_none(forest_binary_classification_with_test):
    """Callable metric when metric_kwargs is None."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def custom_metric(y_true, y_pred):
        return 0.75

    display = report.metrics.summarize(metric=custom_metric, response_method="predict")

    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 1
    assert display.data["score"].values[0] == 0.75


def test_metric_kwargs_override_scorer(linear_regression_with_test):
    """metric_kwargs overrides the kwargs baked into the scorer.

    Regression test for #2203 follow-up.
    """
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


# pos_label


def test_pos_label(forest_binary_classification_with_test):
    """pos_label collapses per-class metrics to a single row."""
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
    assert display.data["label"].isna().all()
    assert display.data["output"].isna().all()


def test_pos_label_strings(forest_binary_classification_with_test):
    """Binary classification with string labels."""
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
    """Binary classification with boolean labels."""
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
    assert any(label is np.False_ for label in labels)
    assert any(label is np.True_ for label in labels)


def test_pos_label_scorer_names(
    forest_binary_classification_with_test,
):
    """pos_label is dispatched with scikit-learn scorer names."""
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
    """pos_label can be set when creating the report."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    classifier = LogisticRegression().fit(X, y)

    # Without pos_label - should have multiple rows (one per class)
    report = EstimatorReport(classifier, X_test=X, y_test=y)
    display = report.metrics.summarize(metric=metric)
    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 2
    assert set(display.data["label"]) == {"A", "B"}

    # With pos_label="B" - should have single row
    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="B")
    display = report.metrics.summarize(metric=metric)
    assert len(display.data) == 1
    score_B = display.data["score"].values[0]
    assert score_B == pytest.approx(metric_fn(y, classifier.predict(X), pos_label="B"))

    # With pos_label="A" - should have single row
    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="A")
    display = report.metrics.summarize(metric=metric)
    assert len(display.data) == 1
    score_A = display.data["score"].values[0]
    assert score_A == pytest.approx(metric_fn(y, classifier.predict(X), pos_label="A"))


# Cache and data_source


def test_cache(forest_binary_classification_with_test):
    """summarize() results are cached; second call returns the same data."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    with check_cache_changed(report._cache):
        result = report.metrics.summarize()
    assert isinstance(result, MetricsSummaryDisplay)

    with check_cache_unchanged(report._cache):
        result_from_cache = report.metrics.summarize()
    assert_frame_equal(result.data, result_from_cache.data)


def test_data_source_both(forest_binary_classification_data):
    """data_source='both' concatenates train and test results."""
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


def test_sklearn_string_without_neg_prefix(linear_regression_with_test):
    """Metrics passed without 'neg_' prefix produce the same result as with it."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display_without = report.metrics.summarize(metric=["mean_squared_error"])
    display_with = report.metrics.summarize(metric=["neg_mean_squared_error"])

    assert set(display_without.data["metric"]) == set(display_with.data["metric"])
    np.testing.assert_allclose(
        display_without.data["score"].values, display_with.data["score"].values
    )
