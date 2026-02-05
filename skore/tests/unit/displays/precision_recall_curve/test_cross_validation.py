import re

import matplotlib as mpl
import numpy as np
import pytest
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skore import CrossValidationReport
from skore._sklearn._plot import PrecisionRecallCurveDisplay
from skore._utils._testing import check_frame_structure, check_legend_position
from skore._utils._testing import (
    check_precision_recall_curve_display_data as check_display_data,
)


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_binary_classification(
    pyplot, logistic_binary_classification_data, data_source
):
    """Check the attributes and default plotting behaviour of the
    precision-recall curve plot with binary data.
    """
    (estimator, X, y), cv = logistic_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    if data_source == "X_y":
        precision_recall_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        precision_recall_kwargs = {"data_source": data_source}

    display = report.metrics.precision_recall(**precision_recall_kwargs)
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    display.plot()
    assert hasattr(display, "ax_")
    assert hasattr(display, "facet_")
    assert hasattr(display, "figure_")

    pos_label = report.estimator_reports_[0].estimator_.classes_[1]

    ax = display.ax_
    assert isinstance(ax, mpl.axes.Axes)
    legend = ax.get_legend()
    assert legend is not None

    plot_data = display.frame(with_average_precision=True)
    average_precision = plot_data["average_precision"]
    assert (
        legend.get_texts()[0].get_text()
        == f"AP={average_precision.mean():.2f}±{average_precision.std():.2f}"
    )

    assert ax.get_xlabel() == "recall"
    assert ax.get_ylabel() == "precision"
    assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    data_source_title = "external" if data_source == "X_y" else data_source.capitalize()
    assert (
        display.figure_.get_suptitle()
        == f"Precision-Recall Curve for {estimator.__class__.__name__}"
        f"\nPositive label: {pos_label}"
        f"\nData source: {data_source_title} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_multiclass_classification(
    pyplot, logistic_multiclass_classification_data, data_source
):
    """Check the attributes and default plotting behaviour of the precision-recall
    curve plot with multiclass data.
    """
    (estimator, X, y), cv = logistic_multiclass_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    if data_source == "X_y":
        precision_recall_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        precision_recall_kwargs = {"data_source": data_source}

    display = report.metrics.precision_recall(**precision_recall_kwargs)
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    display.plot()
    assert hasattr(display, "facet_")

    class_labels = report.estimator_reports_[0].estimator_.classes_

    ax = display.ax_
    assert isinstance(ax, mpl.axes.Axes)
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]

    for label_idx, class_label in enumerate(class_labels):
        plot_data = display.frame(with_average_precision=True)
        average_precision = plot_data.query(f"label == {class_label}")[
            "average_precision"
        ]
        ap_mean = average_precision.mean()
        ap_std = average_precision.std()
        assert (
            legend_texts[label_idx] == f"{class_label} (AP={ap_mean:.2f}±{ap_std:.2f})"
        )

    assert ax.get_xlabel() == "recall"
    assert ax.get_ylabel() == "precision"
    assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)

    data_source_title = "external" if data_source == "X_y" else data_source.capitalize()
    assert (
        display.figure_.get_suptitle()
        == f"Precision-Recall Curve for {estimator.__class__.__name__}"
        f"\nData source: {data_source_title} set"
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["logistic_binary_classification_data", "logistic_multiclass_classification_data"],
)
def test_wrong_kwargs(pyplot, fixture_name, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `relplot_kwargs` argument."""
    (estimator, X, y), cv = request.getfixturevalue(fixture_name), 3

    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.precision_recall()
    err_msg = "Line2D.set() got an unexpected keyword argument 'invalid'"
    with pytest.raises(AttributeError, match=re.escape(err_msg)):
        display.set_style(relplot_kwargs={"invalid": "value"}).plot()


@pytest.mark.parametrize(
    "fixture_name",
    ["logistic_binary_classification_data", "logistic_multiclass_classification_data"],
)
def test_relplot_kwargs(pyplot, fixture_name, request):
    """Check that we can pass keyword arguments to the precision-recall curve plot."""
    (estimator, X, y), cv = request.getfixturevalue(fixture_name), 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.precision_recall()
    multiclass = "multiclass" in fixture_name

    display.plot()
    assert hasattr(display, "facet_")

    if multiclass:
        display.set_style(relplot_kwargs={"palette": ["red", "blue", "green"]}).plot()
    else:
        display.set_style(relplot_kwargs={"color": "red"}).plot()
    assert hasattr(display, "facet_")


@pytest.mark.parametrize("with_average_precision", [False, True])
def test_frame_binary_classification(
    logistic_binary_classification_data, with_average_precision
):
    """Test the frame method with binary classification data."""
    (estimator, X, y), cv = logistic_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    df = report.metrics.precision_recall().frame(
        with_average_precision=with_average_precision
    )
    expected_index = ["split"]
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["split"].nunique() == cv

    if with_average_precision:
        for (_), group in df.groupby(["split"], observed=True):
            assert group["average_precision"].nunique() == 1


@pytest.mark.parametrize("with_average_precision", [False, True])
def test_frame_multiclass_classification(
    logistic_multiclass_classification_data, with_average_precision
):
    """Test the frame method with multiclass classification data."""
    (estimator, X, y), cv = logistic_multiclass_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    df = report.metrics.precision_recall().frame(
        with_average_precision=with_average_precision
    )
    expected_index = ["split", "label"]
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["split"].nunique() == cv
    assert df["label"].nunique() == len(np.unique(y))

    if with_average_precision:
        for (_, _), group in df.groupby(["split", "label"], observed=True):
            assert group["average_precision"].nunique() == 1


def test_legend(
    pyplot, logistic_binary_classification_data, logistic_multiclass_classification_data
):
    """Check the rendering of the legend for with an `CrossValidationReport`."""
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=5)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="upper center", position="inside")

    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=10)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="upper center", position="inside")

    estimator, X, y = logistic_multiclass_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=5)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="upper center", position="inside")

    estimator = LogisticRegression()
    X, y = make_classification(
        n_samples=1_000,
        n_classes=10,
        n_clusters_per_class=1,
        n_informative=10,
        random_state=42,
    )
    report = CrossValidationReport(estimator, X=X, y=y, splitter=10)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="upper center", position="inside")


def test_binary_classification_constructor(logistic_binary_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = logistic_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.precision_recall()

    index_columns = ["estimator", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator"].unique()[0] == report.estimator_name_
        assert df["split"].nunique() == cv
        assert df["label"].unique() == 1

    assert len(display.average_precision) == cv


def test_multiclass_classification_constructor(logistic_multiclass_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = logistic_multiclass_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.precision_recall()

    index_columns = ["estimator", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator"].unique()[0] == report.estimator_name_
        assert df["split"].unique().tolist() == list(range(cv))
        np.testing.assert_array_equal(df["label"].unique(), np.unique(y))

    assert len(display.average_precision) == len(np.unique(y)) * cv


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        ("logistic_binary_classification_data", ["None", "auto"]),
        ("logistic_multiclass_classification_data", ["None", "auto", "label"]),
    ],
)
def test_invalid_subplot_by(fixture_name, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X=X, y=y, splitter=3)

    display = report.metrics.precision_recall()
    valid_values_str = ", ".join(valid_values)
    err_msg = f"subplot_by must be one of {valid_values_str}. Got 'invalid' instead."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "fixture_name, subplot_by, expected_len",
    [
        ("logistic_binary_classification_data", None, 0),
        ("logistic_multiclass_classification_data", "label", 3),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by, expected_len, request):
    """Check that we can pass `None` to `subplot_by`."""
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X=X, y=y, splitter=3)
    display = report.metrics.precision_recall()
    display.plot(subplot_by=subplot_by)
    if subplot_by is None:
        assert isinstance(display.ax_, mpl.axes.Axes)
    else:
        assert len(display.ax_) == expected_len
