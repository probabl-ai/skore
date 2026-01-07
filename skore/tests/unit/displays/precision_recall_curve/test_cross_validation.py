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

    pos_label = report.estimator_reports_[0].estimator_.classes_[1]

    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == cv

    expected_color = sns.color_palette()[:1][0]
    for line in display.lines_:
        assert line.get_color() == expected_color

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

    class_labels = report.estimator_reports_[0].estimator_.classes_

    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(class_labels) * cv

    expected_color = sns.color_palette()[:1][0]
    for line in display.lines_:
        assert line.get_color() == expected_color

    assert len(display.ax_) == len(class_labels)

    for class_label in class_labels:
        ax = display.ax_[class_label]
        assert isinstance(ax, mpl.axes.Axes)
        legend = ax.get_legend()
        assert legend is not None

        plot_data = display.frame(with_average_precision=True)
        average_precision = plot_data.query(f"label == {class_label}")[
            "average_precision"
        ]
        assert (
            legend.get_texts()[0].get_text()
            == f"AP={average_precision.mean():.2f}±{average_precision.std():.2f}"
        )
        assert ax.get_xlabel() == "recall"
        assert ax.get_ylabel() in ("precision", "")
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
        display.plot(relplot_kwargs={"invalid": "value"})


@pytest.mark.parametrize(
    "fixture_name",
    ["logistic_binary_classification_data", "logistic_multiclass_classification_data"],
)
def test_relplot_kwargs(pyplot, fixture_name, request):
    """Check that we can pass keyword arguments to the precision-recall curve plot."""
    (estimator, X, y), cv = request.getfixturevalue(fixture_name), 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.precision_recall()

    display.plot()
    default_color = display.lines_[0].get_color()
    assert default_color == sns.color_palette()[:1][0]

    display.plot(relplot_kwargs={"color": "red"})
    for line in display.lines_:
        assert line.get_color() == "red"
        assert mpl.colors.to_rgb(line.get_color()) != default_color

    display.set_style(relplot_kwargs={"color": "blue"}, policy="update")
    display.plot()
    for line in display.lines_:
        assert line.get_color() == "blue"
        assert mpl.colors.to_rgb(line.get_color()) != default_color


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
    check_legend_position(display.ax_[0], loc="upper center", position="inside")

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
    check_legend_position(display.ax_[0], loc="upper center", position="inside")


def test_binary_classification_constructor(logistic_binary_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = logistic_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.precision_recall()

    index_columns = ["estimator_name", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator_name"].unique() == report.estimator_name_
        assert df["split"].nunique() == cv
        assert df["label"].unique() == 1

    assert len(display.average_precision) == cv


def test_multiclass_classification_constructor(logistic_multiclass_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = logistic_multiclass_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.precision_recall()

    index_columns = ["estimator_name", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator_name"].unique() == report.estimator_name_
        assert df["split"].unique().tolist() == list(range(cv))
        np.testing.assert_array_equal(df["label"].unique(), np.unique(y))

    assert len(display.average_precision) == len(np.unique(y)) * cv


@pytest.mark.parametrize(
    "fixture_name",
    [
        "logistic_binary_classification_data",
        "logistic_multiclass_classification_data",
    ],
)
def test_invalid_subplot_by(fixture_name, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    (estimator, X, y), cv = request.getfixturevalue(fixture_name), 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)

    display = report.metrics.precision_recall()
    valid_values = ["'auto'"]
    if len(np.unique(y)) > 2:
        valid_values.append("'label'")
    else:
        valid_values.append("None")
    valid_values_str = ", ".join(valid_values)
    err_msg = f"subplot_by must be one of {valid_values_str}, got 'invalid' instead."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")
