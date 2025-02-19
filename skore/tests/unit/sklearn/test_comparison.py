import re
from io import BytesIO

import joblib
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from skore import ComparisonReport, EstimatorReport


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

    with pytest.raises(TypeError, match="Expected reports to be an iterable"):
        ComparisonReport(estimator_report)

    with pytest.raises(
        ValueError, match="At least 2 instances of EstimatorReport are needed"
    ):
        ComparisonReport([estimator_report])

    with pytest.raises(TypeError, match="Expected instances of EstimatorReport"):
        ComparisonReport([None, estimator_report])


def test_comparison_report_without_testing_data(binary_classification_model):
    """If there is no test data (`None`) for some estimator report,
    initialization works, but computing metrics can fail.
    """
    estimator, _, _, _, _ = binary_classification_model
    estimator_report = EstimatorReport(estimator, fit=False)

    report = ComparisonReport([estimator_report, estimator_report])

    with pytest.raises(ValueError, match="No test data"):
        report.metrics.report_metrics(data_source="test")


def test_comparison_report_different_test_data(binary_classification_model):
    """Raise an error if the passed estimators do not have the same testing data."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_model
    estimator.fit(X_train, y_train)

    # The estimators that have testing data, need to have the same testing data
    # The estimators that do not have testing data do not count
    with pytest.raises(
        ValueError, match="Expected all estimators to have the same testing data"
    ):
        ComparisonReport(
            [
                EstimatorReport(estimator, X_test=X_test, y_test=y_test),
                EstimatorReport(estimator, X_test=X_test[1:], y_test=y_test[1:]),
            ]
        )

    # The estimators without testing data (i.e. no X_test and no y_test) do not count
    ComparisonReport(
        [
            EstimatorReport(estimator, X_test=X_test, y_test=y_test),
            EstimatorReport(estimator, X_test=X_test, y_test=y_test),
            EstimatorReport(estimator),
        ]
    )

    # If there is an X_test but no y_test, it counts
    with pytest.raises(
        ValueError, match="Expected all estimators to have the same testing data"
    ):
        ComparisonReport(
            [
                EstimatorReport(estimator, fit=False, X_test=X_test, y_test=y_test),
                EstimatorReport(estimator, fit=False, X_test=X_test),
            ]
        )


def test_comparison_report_init_different_ml_usecases(
    binary_classification_model, regression_model
):
    """Raise an error if the passed estimators do not have the same ML usecase."""
    linear_regression_estimator, _, X_test, _, y_test = regression_model
    linear_regression_report = EstimatorReport(
        linear_regression_estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    logistic_regression_estimator, _, X_test, _, y_test = binary_classification_model
    logistic_regression_report = EstimatorReport(
        logistic_regression_estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    with pytest.raises(
        ValueError, match="Expected all estimators to have the same ML usecase"
    ):
        ComparisonReport([linear_regression_report, logistic_regression_report])


def test_comparison_report_init_with_report_names(binary_classification_model):
    """If the estimators are passed as a dict,
    then the estimator names are the dict keys."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport({"r1": estimator_report, "r2": estimator_report})

    pd.testing.assert_index_equal(
        comp.metrics.accuracy().columns,
        pd.Index(["r1", "r2"], name="Estimator"),
    )


def test_comparison_report_init_without_report_names(binary_classification_model):
    """If the estimators are passed as a list,
    then the estimator names are the estimator class names."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport([estimator_report, estimator_report])

    pd.testing.assert_index_equal(
        comp.metrics.accuracy().columns,
        pd.Index(["LogisticRegression", "LogisticRegression"], name="Estimator"),
    )


def test_comparison_report_non_string_report_names(binary_classification_model):
    """If the estimators are passed as a dict with non-string keys,
    then the estimator names are the dict keys converted to strings."""
    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    report = ComparisonReport({0: estimator_report, "1": estimator_report})
    assert report.report_names_ == ["0", "1"]


def test_comparison_report_help(capsys, binary_classification_model):
    """Check the help menu works."""
    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    ComparisonReport([estimator_report, estimator_report]).help()

    captured = capsys.readouterr()
    assert "Tools to compare estimators" in captured.out

    # Check that we have a line with accuracy and the arrow associated with it
    assert re.search(
        r"\.accuracy\([^)]*\).*\(↗︎\).*-.*accuracy", captured.out, re.MULTILINE
    )


def test_comparison_report_repr(binary_classification_model):
    """Check the `__repr__` works."""
    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    repr_str = repr(ComparisonReport([estimator_report, estimator_report]))

    assert "ComparisonReport" in repr_str


def test_comparison_report_pickle(tmp_path, binary_classification_model):
    """Check that we can pickle a comparison report."""
    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    with BytesIO() as stream:
        joblib.dump(ComparisonReport([estimator_report, estimator_report]), stream)


def test_comparison_report_metrics_help(capsys, binary_classification_model):
    """Check that the help method writes to the console."""
    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport([estimator_report, estimator_report])

    report.metrics.help()
    captured = capsys.readouterr()
    assert "Available metrics methods" in captured.out


def test_comparison_report_metrics_repr(binary_classification_model):
    """Check the repr method."""
    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport([estimator_report, estimator_report])

    repr_str = repr(report.metrics)
    assert "skore.ComparisonReport.metrics" in repr_str
    assert "report.metrics.help()" in repr_str


@pytest.mark.parametrize("data_source", ["test", "X_y"])
@pytest.mark.parametrize(
    "metric_name, expected",
    [
        (
            "accuracy",
            pd.DataFrame(
                [[1.0, 1.0]],
                columns=pd.Index(
                    ["LogisticRegression", "LogisticRegression"],
                    name="Estimator",
                ),
                index=pd.Index(["Accuracy (↗︎)"], dtype="object", name="Metric"),
            ),
        ),
        (
            "precision",
            pd.DataFrame(
                [[1.0, 1.0], [1.0, 1.0]],
                columns=pd.Index(
                    ["LogisticRegression", "LogisticRegression"],
                    name="Estimator",
                ),
                index=pd.MultiIndex.from_tuples(
                    [("Precision (↗︎)", 0), ("Precision (↗︎)", 1)],
                    names=["Metric", "Label / Average"],
                ),
            ),
        ),
        (
            "recall",
            pd.DataFrame(
                [[1.0, 1.0], [1.0, 1.0]],
                columns=pd.Index(
                    ["LogisticRegression", "LogisticRegression"],
                    name="Estimator",
                ),
                index=pd.MultiIndex.from_tuples(
                    [("Recall (↗︎)", 0), ("Recall (↗︎)", 1)],
                    names=["Metric", "Label / Average"],
                ),
            ),
        ),
        (
            "brier_score",
            pd.DataFrame(
                [[0.026684, 0.026684]],
                columns=pd.Index(
                    ["LogisticRegression", "LogisticRegression"],
                    name="Estimator",
                ),
                index=pd.Index(["Brier score (↘︎)"], dtype="object", name="Metric"),
            ),
        ),
        (
            "roc_auc",
            pd.DataFrame(
                [[1.0, 1.0]],
                columns=pd.Index(
                    ["LogisticRegression", "LogisticRegression"],
                    name="Estimator",
                ),
                index=pd.Index(["ROC AUC (↗︎)"], dtype="object", name="Metric"),
            ),
        ),
        (
            "log_loss",
            pd.DataFrame(
                [[0.113233, 0.113233]],
                columns=pd.Index(
                    ["LogisticRegression", "LogisticRegression"],
                    name="Estimator",
                ),
                index=pd.Index(["Log loss (↘︎)"], dtype="object", name="Metric"),
            ),
        ),
    ],
)
def test_comparison_report_metrics_binary_classification(
    metric_name, expected, data_source, binary_classification_model
):
    """Check the metrics work."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport([estimator_report, estimator_report])

    # ensure metric is valid
    if data_source == "X_y":
        result = getattr(comp.metrics, metric_name)(
            data_source=data_source, X=X_test, y=y_test
        )
    else:
        result = getattr(comp.metrics, metric_name)(data_source=data_source)
    pd.testing.assert_frame_equal(result, expected)

    # ensure metric is valid even from the cache
    if data_source == "X_y":
        result = getattr(comp.metrics, metric_name)(
            data_source=data_source, X=X_test, y=y_test
        )
    else:
        result = getattr(comp.metrics, metric_name)(data_source=data_source)
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
                    ["LinearRegression", "LinearRegression"],
                    name="Estimator",
                ),
                index=pd.Index(["RMSE (↘︎)"], dtype="object", name="Metric"),
            ),
        ),
        (
            "r2",
            pd.DataFrame(
                [[0.680319, 0.680319]],
                columns=pd.Index(
                    ["LinearRegression", "LinearRegression"],
                    name="Estimator",
                ),
                index=pd.Index(["R² (↗︎)"], dtype="object", name="Metric"),
            ),
        ),
    ],
)
def test_comparison_report_metrics_linear_regression(
    metric_name, expected, data_source, regression_model
):
    """Check the metrics work."""
    estimator, X_train, X_test, y_train, y_test = regression_model
    estimator_report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport([estimator_report, estimator_report])

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


def test_comparison_report_report_metrics_X_y(binary_classification_model):
    """Check that `report_metrics` works with an "X_y" data source."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport([estimator_report, estimator_report])

    result = comp.metrics.report_metrics(
        data_source="X_y",
        X=X_train[:10],
        y=y_train[:10],
    )

    expected = pd.DataFrame(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.01514976, 0.01514976],
        ],
        columns=pd.Index(
            ["LogisticRegression", "LogisticRegression"],
            name="Estimator",
        ),
        index=pd.MultiIndex.from_tuples(
            [
                ("Precision (↗︎)", 0),
                ("Precision (↗︎)", 1),
                ("Recall (↗︎)", 0),
                ("Recall (↗︎)", 1),
                ("ROC AUC (↗︎)", ""),
                ("Brier score (↘︎)", ""),
            ],
            names=["Metric", "Label / Average"],
        ),
    )
    pd.testing.assert_frame_equal(result, expected)

    assert len(comp._cache) == 1
    cached_result = list(comp._cache.values())[0]
    pd.testing.assert_frame_equal(cached_result, expected)


def test_comparison_report_custom_metric_X_y(binary_classification_model):
    """Check that `custom_metric` works with an "X_y" data source."""
    from sklearn.metrics import mean_absolute_error

    estimator, X_train, X_test, y_train, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport([estimator_report, estimator_report])

    expected = pd.DataFrame(
        [[0.0, 0.0]],
        columns=pd.Index(
            ["LogisticRegression", "LogisticRegression"], name="Estimator"
        ),
        index=pd.Index(["MAE (↗︎)"], name="Metric"),
    )

    # ensure metric is valid
    result = comp.metrics.custom_metric(
        metric_function=mean_absolute_error,
        response_method="predict",
        metric_name="MAE (↗︎)",
        data_source="X_y",
        X=X_test,
        y=y_test,
    )
    pd.testing.assert_frame_equal(result, expected)

    # ensure metric is valid even from the cache
    result = comp.metrics.custom_metric(
        metric_function=mean_absolute_error,
        response_method="predict",
        metric_name="MAE (↗︎)",
        data_source="X_y",
        X=X_test,
        y=y_test,
    )
    pd.testing.assert_frame_equal(result, expected)
