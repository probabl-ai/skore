import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.utils._testing import _convert_container

from skore import EstimatorReport, PermutationImportanceDisplay
from skore._utils._testing import custom_r2_score


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
    display = report.inspection.permutation_importance(
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
        "split",
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

    display.plot(metric="accuracy")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_xlabel() == "Decrease in accuracy"
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
    display = report.inspection.permutation_importance(
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
        "split",
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

    display.plot(metric="precision")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    ax = (
        display.ax_.flatten()[0] if isinstance(display.ax_, np.ndarray) else display.ax_
    )
    assert isinstance(ax, mpl.axes.Axes)
    assert ax.get_xlabel() == "Decrease in precision"
    assert ax.get_ylabel() == ""
    legend_labels = [text.get_text() for text in display.facet_.legend.get_texts()]
    assert legend_labels == [str(label) for label in report.estimator_.classes_]
    display.plot(metric="recall")
    ax = (
        display.ax_.flatten()[0] if isinstance(display.ax_, np.ndarray) else display.ax_
    )
    assert ax.get_xlabel() == "Decrease in recall"
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
    display = report.inspection.permutation_importance(
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
        "split",
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

    display.plot(metric="r2")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_xlabel() == "Decrease in r2"
    assert display.ax_.get_ylabel() == ""
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {estimator_name} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_callable_metric(
    pyplot,
    linear_regression_with_train_test,
    data_source,
):
    """Test that callable metrics are properly formatted in xlabel (underscores
    replaced with spaces)."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=custom_r2_score
    )
    display.plot(metric="custom r2 score")
    assert display.ax_.get_xlabel() == "Decrease in custom r2 score"


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
    display = report.inspection.permutation_importance(
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
        "split",
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

    display.plot(metric="r2")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    ax = (
        display.ax_.flatten()[0] if isinstance(display.ax_, np.ndarray) else display.ax_
    )
    assert isinstance(ax, mpl.axes.Axes)
    assert ax.get_xlabel() == "Decrease in r2"
    legend_labels = [text.get_text() for text in display.facet_.legend.get_texts()]
    assert legend_labels == [str(output) for output in range(y_train.shape[1])]
    display.plot(metric="mse")
    ax = (
        display.ax_.flatten()[0] if isinstance(display.ax_, np.ndarray) else display.ax_
    )
    assert ax.get_xlabel() == "Decrease in mse"
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {estimator_name} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_None_single_metric_single_value(
    pyplot,
    logistic_binary_classification_with_train_test,
    data_source,
):
    """Check the behaviour of `subplot_by=None` with a single metric returning
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

    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source
    )
    display.plot(metric="accuracy", subplot_by=None)
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert len(display.facet_.legend.get_texts()) == 0
    assert display.ax_.get_xlabel() == "Decrease in accuracy"
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_None_single_metric_multiple_labels(
    pyplot,
    logistic_multiclass_classification_with_train_test,
    data_source,
):
    """Check the behaviour of `subplot_by=None` with a single metric returning
    multiple values grouped by label."""
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

    metric = make_scorer(precision_score, average=None)
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="precision score", subplot_by=None)
    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.facet_.legend
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == [str(label) for label in report.estimator_.classes_]
    assert display.ax_.get_xlabel() == "Decrease in precision score"


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_None_multiple_metrics_single_value(
    pyplot,
    logistic_binary_classification_with_train_test,
    data_source,
):
    """Check the behaviour of `subplot_by=None` with multiple metrics - plotting
    one metric at a time shows a single plot per metric."""
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

    metric = ["precision", "recall"]
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="precision", subplot_by=None)
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert display.ax_.get_xlabel() == "Decrease in precision"
    display.plot(metric="recall", subplot_by=None)
    assert display.ax_.get_xlabel() == "Decrease in recall"
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_None_single_metric_multiple_outputs(
    pyplot,
    linear_regression_multioutput_with_train_test,
    data_source,
):
    """Check the behaviour of `subplot_by=None` with a single metric returning
    multiple values grouped by output."""
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
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="r2 score", subplot_by=None)
    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.facet_.legend
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == [str(output) for output in range(y_train.shape[1])]
    assert display.ax_.get_xlabel() == "Decrease in r2 score"
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_None_multiple_metrics_multiple_outputs(
    pyplot,
    linear_regression_multioutput_with_train_test,
    data_source,
):
    """Check that with multiple metrics in report, plotting a single metric with
    multiple outputs uses hue=output and subplot_by=None works."""
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

    metric = {
        "r2": make_scorer(r2_score, multioutput="raw_values"),
        "mse": make_scorer(mean_squared_error, multioutput="raw_values"),
    }
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="r2", subplot_by=None)
    assert isinstance(display.ax_, mpl.axes.Axes)
    legend_labels = [text.get_text() for text in display.facet_.legend.get_texts()]
    assert legend_labels == [str(output) for output in range(y_train.shape[1])]
    assert display.ax_.get_xlabel() == "Decrease in r2"


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_auto_single_metric_single_target_classification(
    pyplot,
    logistic_binary_classification_with_train_test,
    data_source,
):
    """Check the behaviour of `subplot_by="auto"` with a single metric and
    single target for classification (no hue, no col, no row)."""
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
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source
    )
    display.plot(metric="accuracy", subplot_by="auto")
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert len(display.facet_.legend.get_texts()) == 0
    assert display.ax_.get_xlabel() == "Decrease in accuracy"
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_auto_single_metric_single_target_regression(
    pyplot,
    linear_regression_with_train_test,
    data_source,
):
    """Check the behaviour of `subplot_by="auto"` with a single metric and
    single target for regression (no hue, no col, no row)."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source
    )
    display.plot(metric="r2", subplot_by="auto")
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert len(display.facet_.legend.get_texts()) == 0
    assert display.ax_.get_xlabel() == "Decrease in r2"
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_auto_multiple_metrics_single_target_classification(
    pyplot,
    logistic_binary_classification_with_train_test,
    data_source,
):
    """Check the behaviour of `subplot_by="auto"` with multiple metrics and
    single target for classification (col=metric, no hue, no row)."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = {
        "precision": make_scorer(precision_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
    }
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="precision", subplot_by="auto")
    ax = (
        display.ax_.flatten()[0] if isinstance(display.ax_, np.ndarray) else display.ax_
    )
    assert isinstance(ax, mpl.axes.Axes)
    assert ax.get_xlabel() == "Decrease in precision"
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_auto_multiple_metrics_single_target_regression(
    pyplot,
    linear_regression_with_train_test,
    data_source,
):
    """Check the behaviour of `subplot_by="auto"` with multiple metrics and
    single target for regression (col=metric, no hue, no row)."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = {
        "r2": make_scorer(r2_score),
        "mse": make_scorer(mean_squared_error),
    }
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="r2", subplot_by="auto")
    ax = (
        display.ax_.flatten()[0] if isinstance(display.ax_, np.ndarray) else display.ax_
    )
    assert isinstance(ax, mpl.axes.Axes)
    assert ax.get_xlabel() == "Decrease in r2"
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_auto_single_metric_multiple_labels(
    pyplot,
    logistic_multiclass_classification_with_train_test,
    data_source,
):
    """Check the behaviour of `subplot_by="auto"` with a single metric and
    multiple labels (hue=label, no col, no row)."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = make_scorer(precision_score, average=None)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="precision score", subplot_by="auto")
    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.facet_.legend
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == [str(label) for label in report.estimator_.classes_]
    assert display.ax_.get_xlabel() == "Decrease in precision score"
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_auto_single_metric_multiple_outputs(
    pyplot,
    linear_regression_multioutput_with_train_test,
    data_source,
):
    """Check the behaviour of `subplot_by="auto"` with a single metric and
    multiple outputs (hue=output, no col, no row)."""
    estimator, X_train, X_test, y_train, y_test = (
        linear_regression_multioutput_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = make_scorer(r2_score, multioutput="raw_values")
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="r2 score", subplot_by="auto")
    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.facet_.legend
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == [str(output) for output in range(y_train.shape[1])]
    assert display.ax_.get_xlabel() == "Decrease in r2 score"
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_auto_multiple_metrics_multiple_labels(
    pyplot,
    logistic_multiclass_classification_with_train_test,
    data_source,
):
    """Check the behaviour of `subplot_by="auto"` with single metric and
    multiple labels (hue=label, no col, no row)."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
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
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="precision", subplot_by="auto")
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert display.ax_.get_xlabel() == "Decrease in precision"
    legend = display.facet_.legend
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == [str(label) for label in report.estimator_.classes_]
    display.plot(metric="recall", subplot_by="auto")
    assert display.ax_.get_xlabel() == "Decrease in recall"
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_auto_multiple_metrics_multiple_outputs(
    pyplot,
    linear_regression_multioutput_with_train_test,
    data_source,
):
    """Check the behaviour of `subplot_by="auto"` with single metric and
    multiple outputs (hue=output, no col, no row)."""
    estimator, X_train, X_test, y_train, y_test = (
        linear_regression_multioutput_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = {
        "r2": make_scorer(r2_score, multioutput="raw_values"),
        "mse": make_scorer(mean_squared_error, multioutput="raw_values"),
    }
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="r2", subplot_by="auto")
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert display.ax_.get_xlabel() == "Decrease in r2"
    legend = display.facet_.legend
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == [str(output) for output in range(y_train.shape[1])]
    display.plot(metric="mse", subplot_by="auto")
    assert display.ax_.get_xlabel() == "Decrease in mse"
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


def test_subplot_by_invalid_column_raises_error(
    pyplot,
    linear_regression_with_train_test,
):
    """Check that using an invalid column name for `subplot_by` raises an error."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = {
        "r2": make_scorer(r2_score),
        "mse": make_scorer(mean_squared_error),
    }
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(n_repeats=2, metric=metric)
    err_msg = (
        "The column\\(s\\) \\['label'\\] are not available\\. You can use the "
        "following values to create subplots:"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(metric="r2", subplot_by="label")


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_string_multiple_metrics_multiclass_with_remaining_column(
    pyplot,
    logistic_multiclass_classification_with_train_test,
    data_source,
):
    """Check that when subplot_by is a string with label and metric is specified,
    it shows one metric with hue=label (multiclass case)."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
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
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="precision", subplot_by="label")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_.flatten()) == len(report.estimator_.classes_)
    for ax in display.ax_.flatten():
        assert isinstance(ax, mpl.axes.Axes)
    legend = display.facet_.legend
    assert legend is not None
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_string_multiple_metrics_binary_class_no_remaining_column(
    pyplot,
    logistic_binary_classification_with_train_test,
    data_source,
):
    """Check that when subplot_by is a string with no remaining column,
    hue is None (single metric + binary class case)."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = {
        "precision": make_scorer(precision_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
    }
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="precision")
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert display.ax_.get_xlabel() == "Decrease in precision"
    assert len(display.facet_.legend.get_texts()) == 0
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


def test_subplot_by_invalid_tuple_columns_raises_error(
    pyplot,
    linear_regression_with_train_test,
):
    """Check that using a tuple with invalid column names for `subplot_by` raises an
    error."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = {
        "r2": make_scorer(r2_score),
        "mse": make_scorer(mean_squared_error),
    }
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(n_repeats=2, metric=metric)
    err_msg = (
        r"The column\(s\) \['metric', 'label'\] are not available\. You can use the "
        r"following values to create subplots:"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(metric="r2", subplot_by=("metric", "label"))


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_subplot_by_string_label_creates_column_subplots(
    pyplot,
    logistic_multiclass_classification_with_train_test,
    data_source,
):
    """Check that subplot_by with label creates a column of subplots."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = make_scorer(precision_score, average=None)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )
    display.plot(metric="precision score", subplot_by="label")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_.flatten()) == len(report.estimator_.classes_)
    for ax in display.ax_.flatten():
        assert isinstance(ax, mpl.axes.Axes)
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance of {report.estimator_name_} on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
@pytest.mark.parametrize(
    "aggregate, expected_value_columns",
    [
        ("mean", ["value"]),
        ("std", ["value"]),
        (("mean", "std"), ["value_mean", "value_std"]),
        (("min", "max"), ["value_min", "value_max"]),
    ],
)
def test_frame_aggregate_parameter(
    linear_regression_with_train_test,
    data_source,
    aggregate,
    expected_value_columns,
):
    """Check that the aggregate parameter correctly affects the column names."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    n_repeats = 2
    display = report.inspection.permutation_importance(
        n_repeats=n_repeats, data_source=data_source
    )

    df = display.frame(aggregate=aggregate)
    base_columns = ["data_source", "metric", "feature"]
    expected_columns = base_columns + expected_value_columns
    assert sorted(df.columns.tolist()) == sorted(expected_columns)


@pytest.mark.parametrize("data_source", ["train", "test"])
@pytest.mark.parametrize(
    "metric_filter, expected_metrics",
    [
        ("r2", ["r2"]),
        (["r2"], ["r2"]),
        (["r2", "mse"], ["r2", "mse"]),
    ],
)
def test_frame_metric_parameter(
    linear_regression_with_train_test,
    data_source,
    metric_filter,
    expected_metrics,
):
    """Check that the metric parameter correctly filters the output dataframe."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = {
        "r2": make_scorer(r2_score),
        "mse": make_scorer(mean_squared_error),
    }
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    n_repeats = 2
    display = report.inspection.permutation_importance(
        n_repeats=n_repeats, data_source=data_source, metric=metric
    )

    df_all = display.frame()
    assert set(df_all["metric"].unique()) == {"r2", "mse"}

    df_filtered = display.frame(metric=metric_filter)
    assert set(df_filtered["metric"].unique()) == set(expected_metrics)


@pytest.mark.parametrize("data_source", ["train", "test"])
@pytest.mark.parametrize("aggregate", [None, ("mean", "std")])
def test_frame_mixed_averaged_and_non_averaged_metrics(
    logistic_binary_classification_with_train_test,
    data_source,
    aggregate,
):
    """Check that mixed averaged and non-averaged metrics are handled correctly.

    Averaged metrics (like accuracy) should have NaN in label/output columns,
    while non-averaged metrics (like precision/recall with average=None) should
    have actual values in label columns.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = {
        "accuracy": "accuracy",  # averaged metric
        "precision": make_scorer(precision_score, average=None),  # non-averaged
        "recall": make_scorer(recall_score, average=None),  # non-averaged
    }
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    n_repeats = 2
    display = report.inspection.permutation_importance(
        n_repeats=n_repeats, data_source=data_source, metric=metric
    )

    df = display.frame(aggregate=aggregate)

    assert set(df["metric"].unique()) == {"accuracy", "precision", "recall"}
    assert "output" not in df.columns
    assert "label" in df.columns

    df_accuracy = df[df["metric"] == "accuracy"]
    assert df_accuracy["label"].isna().all()

    df_precision = df[df["metric"] == "precision"]
    df_recall = df[df["metric"] == "recall"]
    assert not df_precision["label"].isna().any()
    assert not df_recall["label"].isna().any()
    np.testing.assert_array_equal(
        df_precision["label"].unique(), report.estimator_.classes_
    )
    np.testing.assert_array_equal(
        df_recall["label"].unique(), report.estimator_.classes_
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_plot_mixed_averaged_and_non_averaged_metrics_classification_raises_error(
    pyplot,
    logistic_binary_classification_with_train_test,
    data_source,
):
    """Check that plotting mixed averaged and non-averaged metrics raises an error.

    Then verify that filtering by metric allows plotting to work for classification.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average=None),
    }
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )

    display.plot(metric="accuracy")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")

    display.plot(metric="precision")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_plot_mixed_averaged_and_non_averaged_metrics_regression_raises_error(
    pyplot,
    linear_regression_multioutput_with_train_test,
    data_source,
):
    """Check that plotting mixed averaged and non-averaged metrics raises an error.

    Then verify that filtering by metric allows plotting to work for regression.
    """
    estimator, X_train, X_test, y_train, y_test = (
        linear_regression_multioutput_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator)
    metric = {
        "r2": make_scorer(r2_score),
        "r2_raw": make_scorer(r2_score, multioutput="raw_values"),
    }
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric
    )

    display.plot(metric="r2")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")

    display.plot(metric="r2_raw")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
