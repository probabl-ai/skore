import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import make_scorer, precision_score
from sklearn.utils._testing import _convert_container

from skore import CrossValidationReport, PermutationImportanceDisplay


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_binary_classification_averaged_metrics(
    pyplot,
    cross_validation_report_binary_classification,
    data_source,
):
    """Check the attributes and default plotting behaviour of the permutation
    importance plot with cross-validation, binary classification and averaged metrics.
    """
    report = cross_validation_report_binary_classification
    splitter = len(report.estimator_reports_)
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, seed=0
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
    assert df["split"].nunique() == splitter
    assert df["estimator"].nunique() == 1

    df_frame = display.frame(metric="accuracy")
    expected_frame_columns = [
        "data_source",
        "metric",
        "split",
        "feature",
        "value_mean",
        "value_std",
    ]
    assert sorted(df_frame.columns.tolist()) == sorted(expected_frame_columns)

    display.plot(metric="accuracy")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert display.ax_.get_xlabel() == "Decrease in accuracy"
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance averaged over splits \nof {estimator_name} "
        f"on {data_source} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_single_output_regression(
    pyplot,
    linear_regression_data,
    data_source,
):
    """Check the attributes and default plotting behaviour of the permutation
    importance plot with cross-validation and single output regression."""
    estimator, X, y = linear_regression_data
    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, "dataframe", columns_name=columns_names)

    report = CrossValidationReport(clone(estimator), X, y, splitter=2)
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, seed=0
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
    assert df["metric"].unique() == ["r2"]
    assert df["split"].nunique() == 2

    df_frame = display.frame(metric="r2")
    expected_frame_columns = [
        "data_source",
        "metric",
        "split",
        "feature",
        "value_mean",
        "value_std",
    ]
    assert sorted(df_frame.columns.tolist()) == sorted(expected_frame_columns)

    display.plot(metric="r2")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert display.ax_.get_xlabel() == "Decrease in r2"
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance averaged over splits \nof {estimator_name} "
        f"on {data_source} set"
    )


def test_subplot_by_split(
    pyplot,
    cross_validation_report_binary_classification,
    data_source,
):
    """Check that subplot_by='split' creates one subplot per fold."""
    report = cross_validation_report_binary_classification
    splitter = len(report.estimator_reports_)

    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, seed=0
    )
    display.plot(metric="accuracy", subplot_by="split")

    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_.flatten()) == splitter
    for ax, split_idx in zip(display.ax_.flatten(), range(splitter), strict=True):
        assert isinstance(ax, mpl.axes.Axes)
        assert f"split = {split_idx}" in ax.get_title()
        assert ax.get_xlabel() == "Decrease in accuracy"
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance  \nof {estimator_name} on {data_source} set"
    )


def test_subplot_by_label_aggregates_split_in_remaining(
    pyplot,
    cross_validation_report_multiclass_classification,
    data_source,
):
    """Check that when subplot_by='label' and split is in remaining (not used for
    subplotting), we aggregate over splits and title includes 'averaged over splits'."""
    report = cross_validation_report_multiclass_classification
    metric = make_scorer(precision_score, average=None)
    n_classes = len(report.estimator_reports_[0].estimator_.classes_)

    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric, seed=0
    )
    display.plot(metric="precision score", subplot_by="label")

    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_.flatten()) == n_classes
    for ax in display.ax_.flatten():
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_xlabel() == "Decrease in precision score"
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance averaged over splits \nof {estimator_name} "
        f"on {data_source} set"
    )


def test_subplot_by_tuple_label_split(
    pyplot,
    cross_validation_report_multiclass_classification,
    data_source,
):
    """Check subplot_by=('label', 'split') creates a 2D grid: row=label, col=split."""
    report = cross_validation_report_multiclass_classification
    metric = make_scorer(precision_score, average=None)
    n_classes = len(report.estimator_reports_[0].estimator_.classes_)
    splitter = len(report.estimator_reports_)

    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric, seed=0
    )
    display.plot(metric="precision score", subplot_by=("label", "split"))

    assert isinstance(display.ax_, np.ndarray)
    assert display.ax_.shape == (n_classes, splitter)
    for row_idx in range(n_classes):
        for col_idx in range(splitter):
            ax = display.ax_[row_idx, col_idx]
            assert isinstance(ax, mpl.axes.Axes)
            assert ax.get_xlabel() == "Decrease in precision score"
            assert f"label = {row_idx}" in ax.get_title()
            assert f"split = {col_idx}" in ax.get_title()
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance  \nof {estimator_name} on {data_source} set"
    )


def test_subplot_by_auto_single_metric_multiclass(
    pyplot,
    cross_validation_report_multiclass_classification,
    data_source,
):
    """Check subplot_by='auto' with multiclass: label on columns, split averaged."""
    report = cross_validation_report_multiclass_classification
    metric = make_scorer(precision_score, average=None)

    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, metric=metric, seed=0
    )
    display.plot(metric="precision score", subplot_by="auto")

    n_classes = len(report.estimator_reports_[0].estimator_.classes_)
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_.flatten()) == n_classes
    for ax in display.ax_.flatten():
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_xlabel() == "Decrease in precision score"
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance averaged over splits \nof {estimator_name} "
        f"on {data_source} set"
    )


def test_subplot_by_None_averaged_over_splits(
    pyplot,
    cross_validation_report_binary_classification,
    data_source,
):
    """Check subplot_by=None with only split: averages over splits, single plot."""
    report = cross_validation_report_binary_classification

    display = report.inspection.permutation_importance(
        n_repeats=2, data_source=data_source, seed=0
    )
    display.plot(metric="accuracy", subplot_by=None)

    assert isinstance(display.ax_, mpl.axes.Axes)
    assert len(display.facet_.legend.get_texts()) == 0
    assert display.ax_.get_xlabel() == "Decrease in accuracy"
    estimator_name = display.importances["estimator"].unique()[0]
    assert (
        display.figure_.get_suptitle()
        == f"Permutation importance averaged over splits \nof {estimator_name} "
        f"on {data_source} set"
    )


def test_frame_metric_parameter(
    pyplot,
    linear_regression_data,
    data_source,
):
    """Check that the metric parameter correctly filters the output dataframe."""
    estimator, X, y = linear_regression_data
    X = _convert_container(
        X, "dataframe", columns_name=[f"Feature #{i}" for i in range(X.shape[1])]
    )

    report = CrossValidationReport(clone(estimator), X, y, splitter=2)
    display = report.inspection.permutation_importance(
        n_repeats=2,
        data_source=data_source,
        metric=["r2", "neg_mean_squared_error"],
        seed=0,
    )

    df_all = display.frame()
    assert set(df_all["metric"].unique()) == {"r2", "neg_mean_squared_error"}

    df_filtered = display.frame(metric="r2")
    assert set(df_filtered["metric"].unique()) == {"r2"}


def test_subplot_by_invalid_column_raises_error(
    pyplot,
    cross_validation_report_binary_classification,
):
    """Check that using an invalid column name for subplot_by raises an error."""
    report = cross_validation_report_binary_classification
    display = report.inspection.permutation_importance(
        n_repeats=2, data_source="test", seed=0
    )
    err_msg = (
        r"The column\(s\) \['label'\] are not available\. You can use the "
        r"following values to create subplots:"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(metric="accuracy", subplot_by="label")
