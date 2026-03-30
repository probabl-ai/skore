"""Systematic tests for EstimatorReport.metrics.summarize().

The test matrix covers every combination of:
  - metric input type: default (None), built-in strings, sklearn scorer strings,
    make_scorer instances, plain callables, and mixed lists
  - form: plain list / single value, with extra args (metric_kwargs), as dict
"""

import re

import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import (
    f1_score,
    make_scorer,
    mean_absolute_error,
    precision_score,
    r2_score,
)

from skore import EstimatorReport, MetricsSummaryDisplay

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_display(display, *, expected_metric_names, n_rows=None):
    """Assert basic structure and optional row count of a MetricsSummaryDisplay."""
    assert isinstance(display, MetricsSummaryDisplay)
    assert set(display.data["metric"]) == set(expected_metric_names)
    assert not display.data.empty
    if n_rows is not None:
        assert len(display.data) == n_rows


# ===========================================================================
# 1. Default metrics  (metric=None)
# ===========================================================================


@pytest.mark.parametrize("metric", [None, [], {}])
def test_default_metrics_plain(forest_binary_classification_with_test, metric):
    """metric=None selects the canonical defaults for the ML task."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=metric)

    _assert_display(
        display,
        expected_metric_names={
            "Accuracy",
            "Precision",
            "Recall",
            "ROC AUC",
            "Brier score",
            "Fit time (s)",
            "Predict time (s)",
        },
    )


def test_default_metrics_with_extra_args(linear_regression_multioutput_with_test):
    """metric=None with metric_kwargs alters how the default metrics are computed.

    For multioutput regression, multioutput='raw_values' produces one row per
    output for R² and RMSE.
    """
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric_kwargs={"multioutput": "raw_values"})

    assert isinstance(display, MetricsSummaryDisplay)
    # At least 2 outputs for R² and RMSE respectively
    r2_rows = display.data[display.data["metric"] == "R²"]
    rmse_rows = display.data[display.data["metric"] == "RMSE"]
    assert len(r2_rows) >= 2
    assert len(rmse_rows) >= 2


def test_default_metrics_as_dict(linear_regression_with_test):
    """
    Check that when built-in metric names are passed as a dict,
    the dict keys become the display names.
    """
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric={"My R²": "r2", "My RMSE": "rmse"})

    _assert_display(display, expected_metric_names={"My R²", "My RMSE"}, n_rows=2)


def test_invalid_metric_type(linear_regression_with_test):
    """An integer in the metric list raises a clear ValueError."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    err_msg = re.escape("Invalid type of metric: <class 'int'> for 1")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[1])


# ===========================================================================
# 2. Metric strings
# ===========================================================================


def test_metric_strings_plain(linear_regression_with_test):
    """A list of skore built-in metric strings resolves to correct display names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=["r2", "rmse"])

    _assert_display(
        display,
        expected_metric_names={"R²", "RMSE"},
        n_rows=2,
    )


def test_metric_strings_with_extra_args(forest_binary_classification_with_test):
    """Built-in metric strings with metric_kwargs change how the metric is computed.

    Passing average='macro' collapses per-class precision/recall to a single row.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(
        metric=["precision", "recall"],
        metric_kwargs={"average": "macro"},
    )

    _assert_display(
        display,
        expected_metric_names={"Precision", "Recall"},
        n_rows=2,  # one row per metric when average is a single value
    )
    assert (display.data["average"] == "macro").all()


def test_metric_strings_as_dict(linear_regression_with_test):
    """Metric strings in dict form: dict keys override display names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(
        metric={
            "Custom MAE": "neg_mean_absolute_error",
            "Custom MSE": "neg_mean_squared_error",
        }
    )

    _assert_display(
        display,
        expected_metric_names={"Custom MAE", "Custom MSE"},
        n_rows=2,
    )


@pytest.mark.parametrize("metric", ["public_metric", "_private_metric"])
def test_error_metric_strings(linear_regression_with_test, metric):
    """An unrecognised metric string raises a clear ValueError."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    err_msg = re.escape(f"Invalid metric: {metric!r}.")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[metric])


def test_sklearn_scorer_names_metric_kwargs(forest_binary_classification_with_test):
    """metric_kwargs is not supported when metric is a sklearn scorer name string."""
    classifier, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)

    err_msg = (
        "The `metric_kwargs` parameter is not supported when `metric` is a "
        "scikit-learn scorer name."
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=["f1"], metric_kwargs={"average": "macro"})


# ===========================================================================
# 3. Metric Scorers  (make_scorer / get_scorer instances)
# ===========================================================================


def test_metric_scorers_plain(linear_regression_with_test):
    """A list of sklearn scorers produces the correct metric names and scores."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    r2_scorer = make_scorer(r2_score, response_method="predict")
    mae_scorer = make_scorer(
        mean_absolute_error, response_method="predict", greater_is_better=False
    )

    display = report.metrics.summarize(metric=[r2_scorer, mae_scorer])

    _assert_display(
        display,
        expected_metric_names={"R2 Score", "Mean Absolute Error"},
        n_rows=2,
    )
    scores = display.data.set_index("metric")["score"]
    assert scores["R2 Score"] == pytest.approx(
        r2_score(y_test, estimator.predict(X_test))
    )
    assert scores["Mean Absolute Error"] == pytest.approx(
        mean_absolute_error(y_test, estimator.predict(X_test))
    )


def test_metric_scorers_with_extra_args(forest_binary_classification_with_test):
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

    _assert_display(
        display,
        expected_metric_names={"F1 Score", "Precision Score"},
        n_rows=2,
    )
    averages = display.data.set_index("metric")["average"]
    assert averages["F1 Score"] == "macro"
    assert averages["Precision Score"] == "weighted"


def test_metric_scorers_as_dict(linear_regression_with_test):
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

    _assert_display(
        display,
        expected_metric_names={"Custom R2", "Custom MAE"},
        n_rows=2,
    )


def test_pos_label_scorer_error(forest_binary_classification_with_test):
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


# ===========================================================================
# 4. Metric callables
# ===========================================================================


def test_metric_callables_plain(linear_regression_with_test):
    """A list of plain callables requires response_method in metric_kwargs."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def my_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    display = report.metrics.summarize(
        metric=[my_mae],
        metric_kwargs={"response_method": "predict"},
    )

    _assert_display(display, expected_metric_names={"My Mae"}, n_rows=1)
    # skore won't assume the favorability
    assert set(display.data["favorability"]) == {""}
    score = display.data["score"].values[0]
    assert score == pytest.approx(
        mean_absolute_error(y_test, estimator.predict(X_test))
    )


def test_metric_callables_with_extra_args(linear_regression_with_test):
    """
    Callables that need extra keyword arguments receive them through metric_kwargs.
    """
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    weights = np.ones_like(y_test) * 3.0

    def weighted_mae(y_true, y_pred, weights):
        return float(np.average(np.abs(y_true - y_pred), weights=weights))

    display = report.metrics.summarize(
        metric=[weighted_mae],
        metric_kwargs={"response_method": "predict", "weights": weights},
    )

    _assert_display(display, expected_metric_names={"Weighted Mae"}, n_rows=1)
    score = display.data["score"].values[0]
    assert score == pytest.approx(
        weighted_mae(y_test, estimator.predict(X_test), weights)
    )


def test_metric_callables_as_dict(linear_regression_with_test):
    """Callables in dict form: dict keys become the display names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def my_metric(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    display = report.metrics.summarize(
        metric={"My Custom Error": my_metric},
        metric_kwargs={"response_method": "predict"},
    )

    _assert_display(display, expected_metric_names={"My Custom Error"}, n_rows=1)
    score = display.data["score"].values[0]
    assert score == pytest.approx(
        mean_absolute_error(y_test, estimator.predict(X_test))
    )


def test_custom_metric_no_response_method(forest_binary_classification_with_test):
    """A callable without response_method raises a descriptive ValueError."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def custom_metric(y_true, y_pred):
        return 0.5

    with pytest.raises(ValueError, match="response_method is required"):
        report.metrics.summarize(metric=custom_metric)


# ===========================================================================
# 5. Mix of metric types
# ===========================================================================


def test_metric_mix_plain(linear_regression_with_test):
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

    _assert_display(
        display,
        expected_metric_names={"RMSE", "Mean Absolute Error", "My R2"},
        n_rows=3,
    )


def test_metric_mix_with_extra_args(linear_regression_with_test):
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

    _assert_display(
        display,
        expected_metric_names={"R²", "Mean Absolute Error", "Scaled Mae"},
        n_rows=3,
    )
    scores = display.data.set_index("metric")["score"]
    plain_mae = mean_absolute_error(y_test, estimator.predict(X_test))
    assert scores["Scaled Mae"] == pytest.approx(2.0 * plain_mae)


def test_metric_mix_as_dict(linear_regression_with_test):
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

    _assert_display(
        display,
        expected_metric_names={"Skore RMSE", "Scorer R2", "Callable MAE"},
        n_rows=3,
    )
