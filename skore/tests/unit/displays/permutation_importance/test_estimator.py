import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    r2_score,
    mean_squared_error,
)
from sklearn.utils._testing import _convert_container

from skore import EstimatorReport, PermutationImportanceDisplay


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_binary_classification_averaged_metrics(
    pyplot,
    logistic_binary_classification_with_train_test,
    data_source,
):
    """Check the attributes and default plotting behaviour of the permutation
    importance plot with binary classification data and averaged metrics returning
    a single value."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    n_repeats = 2
    display = report.feature_importance.permutation(
        n_repeats=n_repeats, data_source=data_source
    )
    assert isinstance(display, PermutationImportanceDisplay)

    expected_columns = [
        "estimator",
        "data_source",
        "metric",
        "feature",
        "label",
        "output",
        "repetition",
        "value",
    ]
    df = display.importances
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    for col in ("label", "output"):
        assert df[col].isna().all()
    assert df["data_source"].unique() == [data_source]
    assert df["metric"].unique() == ["accuracy"]
    assert df["estimator"].unique() == [report.estimator_name_]
    assert df["feature"].tolist() == columns_names * n_repeats

    df = display.frame()
    expected_columns = ["data_source", "metric", "feature", "value_mean", "value_std"]
    assert sorted(df.columns.tolist()) == sorted(expected_columns)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_xlabel() == "Decrease of score"
    assert display.ax_.get_ylabel() == ""
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {estimator_name} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_binary_classification_per_label_metrics(
    pyplot,
    logistic_binary_classification_with_train_test,
    data_source,
):
    """Check the attributes and default plotting behaviour of the permutation
    importance plot with binary classification data and per-label metrics returning
    a value for each label."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = {
        "precision": make_scorer(precision_score, average=None),
        "recall": make_scorer(recall_score, average=None),
    }
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    n_repeats = 2
    display = report.feature_importance.permutation(
        n_repeats=n_repeats, data_source=data_source, metric=metric
    )
    assert isinstance(display, PermutationImportanceDisplay)

    expected_columns = [
        "estimator",
        "data_source",
        "metric",
        "feature",
        "label",
        "output",
        "repetition",
        "value",
    ]
    df = display.importances
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    np.testing.assert_array_equal(df["label"].unique(), report.estimator_.classes_)
    assert df["output"].isna().all()
    assert df["data_source"].unique() == [data_source]
    assert df["metric"].unique().tolist() == ["precision", "recall"]
    assert df["estimator"].unique() == [report.estimator_name_]
    assert df["feature"].tolist() == columns_names * n_repeats * len(
        report.estimator_.classes_
    ) * len(metric)

    df = display.frame()
    expected_columns = [
        "data_source",
        "metric",
        "feature",
        "label",
        "value_mean",
        "value_std",
    ]
    assert sorted(df.columns.tolist()) == sorted(expected_columns)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, np.ndarray)
    for ax, metric_name in zip(display.ax_.flatten(), metric.keys(), strict=True):
        assert isinstance(ax, mpl.axes.Axes)

        assert ax.get_xlabel() == "Decrease of score"
        assert ax.get_ylabel() == ""
        assert ax.get_title() == f"metric = {metric_name}"
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {estimator_name} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_single_output_regression(
    pyplot,
    linear_regression_with_train_test,
    data_source,
):
    """Check the attributes and default plotting behaviour of the permutation
    importance plot with binary classification data and averaged metrics returning
    a single value."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    n_repeats = 2
    display = report.feature_importance.permutation(
        n_repeats=n_repeats, data_source=data_source
    )
    assert isinstance(display, PermutationImportanceDisplay)

    expected_columns = [
        "estimator",
        "data_source",
        "metric",
        "feature",
        "label",
        "output",
        "repetition",
        "value",
    ]
    df = display.importances
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    for col in ("label", "output"):
        assert df[col].isna().all()
    assert df["data_source"].unique() == [data_source]
    assert df["metric"].unique() == ["r2"]
    assert df["estimator"].unique() == [report.estimator_name_]
    assert df["feature"].tolist() == columns_names * n_repeats

    df = display.frame()
    expected_columns = ["data_source", "metric", "feature", "value_mean", "value_std"]
    assert sorted(df.columns.tolist()) == sorted(expected_columns)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_xlabel() == "Decrease of score"
    assert display.ax_.get_ylabel() == ""
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {estimator_name} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_multi_output_regression(
    pyplot,
    linear_regression_multioutput_with_train_test,
    data_source,
):
    """Check the attributes and default plotting behaviour of the permutation
    importance plot with binary classification data and averaged metrics returning
    a single value."""
    estimator, X_train, X_test, y_train, y_test = (
        linear_regression_multioutput_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    n_repeats = 2
    metric = {
        "r2": make_scorer(r2_score, multioutput="raw_values"),
        "mse": make_scorer(mean_squared_error, multioutput="raw_values"),
    }
    display = report.feature_importance.permutation(
        n_repeats=n_repeats, data_source=data_source, metric=metric
    )
    assert isinstance(display, PermutationImportanceDisplay)

    expected_columns = [
        "estimator",
        "data_source",
        "metric",
        "feature",
        "label",
        "output",
        "repetition",
        "value",
    ]
    df = display.importances
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    assert df["label"].isna().all()
    assert df["output"].unique().tolist() == list(range(y_train.shape[1]))
    assert df["data_source"].unique() == [data_source]
    assert df["metric"].unique().tolist() == ["r2", "mse"]
    assert df["estimator"].unique() == [report.estimator_name_]
    assert df["feature"].tolist() == columns_names * n_repeats * y_train.shape[1] * len(
        metric
    )

    df = display.frame()
    expected_columns = [
        "data_source",
        "metric",
        "feature",
        "output",
        "value_mean",
        "value_std",
    ]
    assert sorted(df.columns.tolist()) == sorted(expected_columns)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, np.ndarray)
    for ax, metric_name in zip(display.ax_.flatten(), metric.keys(), strict=True):
        assert isinstance(ax, mpl.axes.Axes)

        assert ax.get_xlabel() == "Decrease of score"
        assert ax.get_ylabel() == ""
        assert ax.get_title() == f"metric = {metric_name}"
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {estimator_name} on {data_source} set"
    )


def test_subplot_by_None(
    pyplot,
    logistic_multiclass_classification_with_train_test,
    linear_regression_multioutput_with_train_test,
):
    """Check the behaviour of the `subplot_by=None` with single or multiple
    scores values."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    # case allowing to group by label when a single metric returns multiple values
    metric = make_scorer(precision_score, average=None)
    display = report.feature_importance.permutation(
        n_repeats=2, data_source="train", metric=metric
    )
    display.plot(subplot_by=None)
    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.facet_.legend
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == [str(label) for label in report.estimator_.classes_]

    # case allowing to group by the metric that returns a single value
    metric = {
        "precision": make_scorer(precision_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
    }
    display = report.feature_importance.permutation(
        n_repeats=2, data_source="train", metric=metric
    )
    display.plot(subplot_by=None)
    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.facet_.legend
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == list(metric.keys())

    # case allowing to group by output when a single metric returns multiple values
    estimator, X_train, X_test, y_train, y_test = (
        linear_regression_multioutput_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    metric = make_scorer(r2_score, multioutput="raw_values")
    display = report.feature_importance.permutation(
        n_repeats=2, data_source="train", metric=metric
    )
    display.plot(subplot_by=None)
    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.facet_.legend
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == [str(output) for output in range(y_train.shape[1])]

    # case where we should raise an error because there is too much information to plot
    # when subplot_by is None
    # multiple metrics with multiple-outputs
    metric = {
        "r2": make_scorer(r2_score, multioutput="raw_values"),
        "mse": make_scorer(mean_squared_error, multioutput="raw_values"),
    }
    display = report.feature_importance.permutation(
        n_repeats=2, data_source="train", metric=metric
    )
    err_msg = "Cannot plot all the available information available on a single plot."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by=None)
