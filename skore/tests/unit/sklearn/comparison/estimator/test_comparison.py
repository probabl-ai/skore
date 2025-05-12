import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from skore import ComparisonReport, EstimatorReport
from skore.sklearn._plot.metrics import (
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)


@pytest.fixture
def binary_classification_model():
    """Create a binary classification dataset and return fitted estimator and data."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return LogisticRegression(random_state=42), X_train, X_test, y_train, y_test


@pytest.fixture
def regression_model():
    """Create a binary classification dataset and return fitted estimator and data."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return LinearRegression(), X_train, X_test, y_train, y_test


def test_comparison_report_init_wrong_parameters(binary_classification_model):
    """If the input is not valid, raise."""

    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    with pytest.raises(TypeError, match="Expected reports to be a list or dict"):
        ComparisonReport(estimator_report)

    with pytest.raises(ValueError, match="Expected at least 2 reports to compare"):
        ComparisonReport([estimator_report])

    with pytest.raises(
        TypeError, match="Expected .* EstimatorReport or .* CrossValidationReport"
    ):
        ComparisonReport([None, estimator_report])


def test_comparison_report_without_testing_data(binary_classification_model):
    """If there is no test data (`None`) for some estimator report,
    initialization works, but computing metrics can fail.
    """
    estimator, X_train, _, y_train, _ = binary_classification_model
    estimator_report_1 = EstimatorReport(estimator, X_train=X_train, y_train=y_train)
    estimator_report_2 = EstimatorReport(estimator, X_train=X_train, y_train=y_train)

    report = ComparisonReport([estimator_report_1, estimator_report_2])

    with pytest.raises(ValueError, match="No test data"):
        report.metrics.report_metrics(data_source="test")


def test_comparison_report_different_test_data(binary_classification_model):
    """Raise an error if the passed estimators do not have the same testing targets."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_model
    estimator.fit(X_train, y_train)

    # The estimators that have testing data need to have the same testing targets
    with pytest.raises(
        ValueError, match="Expected all estimators to share the same test targets."
    ):
        ComparisonReport(
            [
                EstimatorReport(estimator, y_test=y_test),
                EstimatorReport(estimator, y_test=y_test[1:]),
            ]
        )

    # The estimators without testing data (i.e., no y_test) do not count
    ComparisonReport(
        [
            EstimatorReport(estimator, X_test=X_test, y_test=y_test),
            EstimatorReport(estimator, X_test=X_test, y_test=y_test),
            EstimatorReport(estimator),
        ]
    )

    # If there is an X_test but no y_test, it should not raise an error
    ComparisonReport(
        [
            EstimatorReport(estimator, fit=False, X_test=X_test, y_test=None),
            EstimatorReport(estimator, fit=False, X_test=X_test, y_test=None),
        ]
    )


@pytest.fixture
def estimator_reports(binary_classification_model):
    logistic_regression_estimator, X_train, X_test, y_train, y_test = (
        binary_classification_model
    )
    estimator_report_1 = EstimatorReport(
        logistic_regression_estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    estimator_report_2 = EstimatorReport(
        logistic_regression_estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    return estimator_report_1, estimator_report_2


def test_comparison_report_init_different_ml_usecases(
    estimator_reports, regression_model
):
    """Raise an error if the passed estimators do not have the same ML usecase."""
    linear_regression_estimator, X_train, X_test, y_train, y_test = regression_model
    linear_regression_report = EstimatorReport(
        linear_regression_estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    logistic_regression_report, _ = estimator_reports

    with pytest.raises(
        ValueError, match="Expected all estimators to have the same ML usecase"
    ):
        ComparisonReport([linear_regression_report, logistic_regression_report])


def test_comparison_report_init_with_report_names(estimator_reports):
    """If the estimators are passed as a dict,
    then the estimator names are the dict keys."""
    estimator_report_1, estimator_report_2 = estimator_reports

    report = ComparisonReport({"r1": estimator_report_1, "r2": estimator_report_2})

    pd.testing.assert_index_equal(
        report.metrics.accuracy().columns,
        pd.Index(["r1", "r2"], name="Estimator"),
    )


def test_comparison_report_init_without_report_names(estimator_reports):
    """If the estimators are passed as a list,
    then the estimator names are the estimator class names."""
    estimator_report_1, estimator_report_2 = estimator_reports

    report = ComparisonReport([estimator_report_1, estimator_report_2])

    pd.testing.assert_index_equal(
        report.metrics.accuracy().columns,
        pd.Index(["LogisticRegression_1", "LogisticRegression_2"], name="Estimator"),
    )


def test_comparison_report_non_string_report_names(estimator_reports):
    """If the reports are passed as a dict, then keys must be strings."""
    estimator_report_1, estimator_report_2 = estimator_reports
    with pytest.raises(TypeError, match="Expected all report names to be strings"):
        ComparisonReport({0: estimator_report_1, "1": estimator_report_2})


@pytest.fixture
def report_regression(regression_model):
    estimator, X_train, X_test, y_train, y_test = regression_model
    estimator_report_1 = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    estimator_report_2 = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    return ComparisonReport([estimator_report_1, estimator_report_2])


@pytest.fixture
def report_classification(estimator_reports):
    estimator_report_1, estimator_report_2 = estimator_reports
    return ComparisonReport([estimator_report_1, estimator_report_2])


@pytest.fixture
def report(report_classification):
    return report_classification


@pytest.mark.parametrize("data_source", ["test", "X_y"])
@pytest.mark.parametrize(
    "metric_name, expected",
    [
        (
            "accuracy",
            pd.DataFrame(
                [[1.0, 1.0]],
                columns=pd.Index(
                    ["LogisticRegression_1", "LogisticRegression_2"],
                    name="Estimator",
                ),
                index=pd.Index(["Accuracy"], dtype="object", name="Metric"),
            ),
        ),
        (
            "precision",
            pd.DataFrame(
                [[1.0, 1.0], [1.0, 1.0]],
                columns=pd.Index(
                    ["LogisticRegression_1", "LogisticRegression_2"],
                    name="Estimator",
                ),
                index=pd.MultiIndex.from_tuples(
                    [("Precision", 0), ("Precision", 1)],
                    names=["Metric", "Label / Average"],
                ),
            ),
        ),
        (
            "recall",
            pd.DataFrame(
                [[1.0, 1.0], [1.0, 1.0]],
                columns=pd.Index(
                    ["LogisticRegression_1", "LogisticRegression_2"],
                    name="Estimator",
                ),
                index=pd.MultiIndex.from_tuples(
                    [("Recall", 0), ("Recall", 1)],
                    names=["Metric", "Label / Average"],
                ),
            ),
        ),
        (
            "brier_score",
            pd.DataFrame(
                [[0.026684, 0.026684]],
                columns=pd.Index(
                    ["LogisticRegression_1", "LogisticRegression_2"],
                    name="Estimator",
                ),
                index=pd.Index(["Brier score"], dtype="object", name="Metric"),
            ),
        ),
        (
            "roc_auc",
            pd.DataFrame(
                [[1.0, 1.0]],
                columns=pd.Index(
                    ["LogisticRegression_1", "LogisticRegression_2"],
                    name="Estimator",
                ),
                index=pd.Index(["ROC AUC"], dtype="object", name="Metric"),
            ),
        ),
        (
            "log_loss",
            pd.DataFrame(
                [[0.113233, 0.113233]],
                columns=pd.Index(
                    ["LogisticRegression_1", "LogisticRegression_2"],
                    name="Estimator",
                ),
                index=pd.Index(["Log loss"], dtype="object", name="Metric"),
            ),
        ),
    ],
)
def test_comparison_report_metrics_binary_classification(
    metric_name, expected, data_source, binary_classification_model, report
):
    """Check the metrics work."""
    _, _, X_test, _, y_test = binary_classification_model

    # ensure metric is valid
    if data_source == "X_y":
        result = getattr(report.metrics, metric_name)(
            data_source=data_source, X=X_test, y=y_test
        )
    else:
        result = getattr(report.metrics, metric_name)(data_source=data_source)
    pd.testing.assert_frame_equal(result, expected)

    # ensure metric is valid even from the cache
    if data_source == "X_y":
        result = getattr(report.metrics, metric_name)(
            data_source=data_source, X=X_test, y=y_test
        )
    else:
        result = getattr(report.metrics, metric_name)(data_source=data_source)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("data_source", ["test", "X_y"])
@pytest.mark.parametrize(
    "metric_name, expected",
    [
        (
            "rmse",
            pd.DataFrame(
                [[0.27699, 0.27699]],
                columns=pd.Index(
                    ["LinearRegression_1", "LinearRegression_2"],
                    name="Estimator",
                ),
                index=pd.Index(["RMSE"], dtype="object", name="Metric"),
            ),
        ),
        (
            "r2",
            pd.DataFrame(
                [[0.680319, 0.680319]],
                columns=pd.Index(
                    ["LinearRegression_1", "LinearRegression_2"],
                    name="Estimator",
                ),
                index=pd.Index(["R²"], dtype="object", name="Metric"),
            ),
        ),
    ],
)
def test_comparison_report_metrics_linear_regression(
    metric_name, expected, data_source, regression_model
):
    """Check the metrics work."""
    estimator, X_train, X_test, y_train, y_test = regression_model
    estimator_report_1 = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    estimator_report_2 = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport([estimator_report_1, estimator_report_2])

    # ensure metric is valid
    if data_source == "X_y":
        result = getattr(comp.metrics, metric_name)(
            data_source=data_source, X=X_test, y=y_test
        )
    else:
        result = getattr(comp.metrics, metric_name)()
    pd.testing.assert_frame_equal(result, expected)

    # ensure metric is valid even from the cache
    if data_source == "X_y":
        result = getattr(comp.metrics, metric_name)(
            data_source=data_source, X=X_test, y=y_test
        )
    else:
        result = getattr(comp.metrics, metric_name)()
    pd.testing.assert_frame_equal(result, expected)


def test_comparison_report_report_metrics_X_y(binary_classification_model, report):
    """Check that `report_metrics` works with an "X_y" data source."""
    _, X_train, _, y_train, _ = binary_classification_model

    result = report.metrics.report_metrics(
        data_source="X_y",
        X=X_train[:10],
        y=y_train[:10],
    )
    assert "Favorability" not in result.columns

    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Precision", 0),
            ("Precision", 1),
            ("Recall", 0),
            ("Recall", 1),
            ("ROC AUC", ""),
            ("Brier score", ""),
            ("Fit time (s)", ""),
            ("Predict time (s)", ""),
        ],
        names=["Metric", "Label / Average"],
    )
    expected_columns = pd.Index(
        ["LogisticRegression_1", "LogisticRegression_2"],
        name="Estimator",
    )

    pd.testing.assert_index_equal(result.index, expected_index)
    pd.testing.assert_index_equal(result.columns, expected_columns)

    assert len(report._cache) == 1
    cached_result = list(report._cache.values())[0]
    pd.testing.assert_index_equal(cached_result.index, expected_index)
    pd.testing.assert_index_equal(cached_result.columns, expected_columns)


def test_comparison_report_custom_metric_X_y(binary_classification_model, report):
    """Check that `custom_metric` works with an "X_y" data source."""
    _, _, X_test, _, y_test = binary_classification_model

    expected = pd.DataFrame(
        [[0.0, 0.0]],
        columns=pd.Index(
            ["LogisticRegression_1", "LogisticRegression_2"], name="Estimator"
        ),
        index=pd.Index(["MAE"], name="Metric"),
    )

    # ensure metric is valid
    result = report.metrics.custom_metric(
        metric_function=mean_absolute_error,
        response_method="predict",
        metric_name="MAE",
        data_source="X_y",
        X=X_test,
        y=y_test,
    )
    pd.testing.assert_frame_equal(result, expected)

    # ensure metric is valid even from the cache
    result = report.metrics.custom_metric(
        metric_function=mean_absolute_error,
        response_method="predict",
        metric_name="MAE",
        data_source="X_y",
        X=X_test,
        y=y_test,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_cross_validation_report_flat_index(binary_classification_model):
    """Check that the index is flattened when `flat_index` is True.

    Since `pos_label` is None, then by default a MultiIndex would be returned.
    Here, we force to have a single-index by passing `flat_index=True`.
    """
    estimator, X_train, X_test, y_train, y_test = binary_classification_model
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result = report.metrics.report_metrics(flat_index=True)
    assert result.shape == (8, 2)
    assert isinstance(result.index, pd.Index)
    assert result.index.tolist() == [
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "brier_score",
        "fit_time_s",
        "predict_time_s",
    ]
    assert result.columns.tolist() == ["report_1", "report_2"]


def test_estimator_report_report_metrics_indicator_favorability(report):
    """Check that the behaviour of `indicator_favorability` is correct."""
    result = report.metrics.report_metrics(indicator_favorability=True)
    assert "Favorability" in result.columns
    indicator = result["Favorability"]
    assert indicator["Precision"].tolist() == ["(↗︎)", "(↗︎)"]
    assert indicator["Recall"].tolist() == ["(↗︎)", "(↗︎)"]
    assert indicator["ROC AUC"].tolist() == ["(↗︎)"]
    assert indicator["Brier score"].tolist() == ["(↘︎)"]


def test_comparison_report_aggregate(report):
    """Passing `aggregate` should have no effect, as this argument is only relevant
    when comparing `CrossValidationReport`s."""
    assert_allclose(
        report.metrics.report_metrics(aggregate="mean"),
        report.metrics.report_metrics(),
    )


@pytest.mark.parametrize("plot_data_source", ["test", "X_y"])
@pytest.mark.parametrize(
    "plot_ml_task, plot_name, plot_cls, plot_attributes",
    [
        (
            "binary_classification",
            "roc",
            RocCurveDisplay,
            {
                "fpr": {1: [[0, 0, 0, 1], [0, 0, 0, 1]]},
                "tpr": {1: [[0, 0.1, 1, 1], [0, 0.1, 1, 1]]},
                "roc_auc": {1: [1, 1]},
            },
        ),
        (
            "binary_classification",
            "precision_recall",
            PrecisionRecallCurveDisplay,
            {
                "precision": {
                    1: [
                        [0.4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0.4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ],
                },
                "recall": {
                    1: [
                        [1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0],
                        [1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0],
                    ]
                },
                "average_precision": {1: [0.99, 0.99]},
            },
        ),
        (
            "regression",
            "prediction_error",
            PredictionErrorDisplay,
            {
                "y_true": [
                    (
                        [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                        + [1, 1, 1, 0]
                    ),
                    (
                        [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                        + [1, 1, 1, 0]
                    ),
                ],
                "y_pred": [
                    (
                        [0.32, 1.25, 0.94, 0.77, -0.58, 0.89, 0.12, 0.51, 0.70, 0.52]
                        + [0.44, 0.14, 0.15, -0.13, -0.27, 0.24, 0.90, 0.22, 0.04]
                        + [-0.18, 0.20, 0.66, 0.99, 0.70, -0.03]
                    ),
                    (
                        [0.32, 1.25, 0.94, 0.77, -0.58, 0.89, 0.12, 0.51, 0.70, 0.52]
                        + [0.44, 0.14, 0.15, -0.13, -0.27, 0.24, 0.90, 0.22, 0.04]
                        + [-0.18, 0.20, 0.66, 0.99, 0.70, -0.03]
                    ),
                ],
            },
        ),
    ],
)
def test_comparison_report_plots(
    plot_data_source,
    plot_ml_task,
    plot_name,
    plot_cls,
    plot_attributes,
    binary_classification_model,
    regression_model,
    report_classification,
    report_regression,
):
    if plot_ml_task == "binary_classification":
        _, _, X_test, _, y_test = binary_classification_model
        comp = report_classification
    else:
        _, _, X_test, _, y_test = regression_model
        comp = report_regression

    if plot_data_source == "X_y":
        arguments = {"data_source": plot_data_source, "X": X_test, "y": y_test}
    else:
        arguments = {"data_source": plot_data_source}

    # Ensure display object is available
    display = getattr(comp.metrics, plot_name)(**arguments)

    # Ensure display object is of good type
    assert isinstance(display, plot_cls)

    # Ensure data source is well set
    assert display.data_source == plot_data_source

    # Ensure all attributes to test are well set
    for attribute, value in plot_attributes.items():
        display_attribute_value = getattr(display, attribute)
        if isinstance(value, dict):
            for k, v in value.items():
                assert isinstance(display_attribute_value, dict)
                assert k in display_attribute_value
                assert_allclose(display_attribute_value[k], v, atol=1e-2)
        elif isinstance(value, list):
            assert_allclose(display_attribute_value, value, atol=1e-2)
        else:
            raise NotImplementedError()

    # Ensure plot is callable
    display.plot()


def test_random_state(report_regression):
    """If random_state is None (the default) the call should not be cached."""
    report_regression.metrics.prediction_error()
    # skore should store the y_pred of the internal estimators, but not the plot
    assert report_regression._cache == {}


@pytest.mark.parametrize("data_source", ["train", "test"])
@pytest.mark.parametrize(
    "response_method", ["predict", "predict_proba", "decision_function"]
)
@pytest.mark.parametrize("pos_label", [None, 0, 1])
def test_comparison_report_get_predictions(
    report, data_source, response_method, pos_label
):
    """Check the behaviour of the `get_predictions` method."""
    predictions = report.get_predictions(
        data_source=data_source, response_method=response_method, pos_label=pos_label
    )
    assert len(predictions) == 2
    for split_idx, split_predictions in enumerate(predictions):
        if data_source == "train":
            expected_shape = report.reports_[split_idx].y_train.shape
        else:
            expected_shape = report.reports_[split_idx].y_test.shape
        assert split_predictions.shape == expected_shape


def test_comparison_report_get_predictions_error(report):
    """Check that we raise an error when the data source is invalid."""
    with pytest.raises(ValueError, match="Invalid data source"):
        report.get_predictions(data_source="invalid", response_method="predict")


def test_comparison_report_timings(report):
    """Check the general behaviour of the `timings` method."""
    timings = report.metrics.timings()
    assert isinstance(timings, pd.DataFrame)
    assert timings.index.tolist() == ["Fit time (s)"]
    assert timings.columns.tolist() == report.report_names_

    report.get_predictions(data_source="train", response_method="predict")
    timings = report.metrics.timings()
    assert isinstance(timings, pd.DataFrame)
    assert timings.index.tolist() == ["Fit time (s)", "Predict time train (s)"]
    assert timings.columns.tolist() == report.report_names_

    report.get_predictions(data_source="test", response_method="predict")
    timings = report.metrics.timings()
    assert isinstance(timings, pd.DataFrame)
    assert timings.index.tolist() == [
        "Fit time (s)",
        "Predict time train (s)",
        "Predict time test (s)",
    ]
    assert timings.columns.tolist() == report.report_names_


def test_comparison_report_timings_flat_index(report):
    """Check that time measurements have _s suffix with flat_index=True."""
    report.get_predictions(data_source="train", response_method="predict")
    report.get_predictions(data_source="test", response_method="predict")

    # Get metrics with flat_index=True
    results = report.metrics.report_metrics(flat_index=True)

    # Check that expected time measurements are in index with _s suffix
    assert "fit_time_s" in results.index
    assert "predict_time_s" in results.index
