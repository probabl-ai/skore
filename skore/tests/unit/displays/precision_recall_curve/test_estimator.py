import re

import matplotlib as mpl
import numpy as np
import pytest
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from skore import EstimatorReport
from skore._sklearn._plot import PrecisionRecallCurveDisplay
from skore._utils._testing import check_frame_structure, check_legend_position
from skore._utils._testing import (
    check_precision_recall_curve_display_data as check_display_data,
)


def test_binary_classification(pyplot, logistic_binary_classification_with_train_test):
    """Check the attributes and default plotting behaviour of the
    precision-recall curve plot with binary data.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    display.plot()
    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == 1
    precision_recall_curve_mpl = display.lines_[0]
    assert isinstance(precision_recall_curve_mpl, mpl.lines.Line2D)

    ax = display.ax_
    assert isinstance(ax, mpl.axes.Axes)
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    plot_data = display.frame(with_average_precision=True)
    average_precision = plot_data["average_precision"].iloc[0]
    assert legend_texts[0] == f"AP={average_precision:.2f}"
    expected_color = sns.color_palette()[:1][0]
    assert precision_recall_curve_mpl.get_color() == expected_color

    assert ax.get_xlabel() == "recall"
    assert ax.get_ylabel() in ("precision", "")
    assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    assert (
        display.figure_.get_suptitle()
        == f"Precision-Recall Curve for {estimator.__class__.__name__}"
        f"\nPositive label: {estimator.classes_[1]}"
        f"\nData source: Test set"
    )


def test_multiclass_classification(
    pyplot, logistic_multiclass_classification_with_train_test
):
    """Check the attributes and default plotting behaviour of the precision-recall
    curve plot with multiclass data.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(estimator.classes_)

    assert isinstance(display.ax_[0], mpl.axes.Axes)
    assert len(display.ax_) == len(estimator.classes_)

    expected_color = sns.color_palette()[:1][0]
    for class_label in estimator.classes_:
        precision_recall_curve_mpl = display.lines_[class_label]
        assert isinstance(precision_recall_curve_mpl, mpl.lines.Line2D)
        ax = display.ax_[class_label]
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]
        plot_data = display.frame(with_average_precision=True)
        average_precision = plot_data.query(f"label == {class_label}")[
            "average_precision"
        ].iloc[0]
        assert legend_texts[0] == f"AP={average_precision:.2f}"
        assert precision_recall_curve_mpl.get_color() == expected_color

        assert ax.get_xlabel() == "recall"
        assert ax.get_ylabel() in ("precision", "")
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)

    assert (
        display.figure_.get_suptitle()
        == f"Precision-Recall Curve for {estimator.__class__.__name__}"
        f"\nData source: Test set"
    )


def test_data_source(pyplot, logistic_binary_classification_with_train_test):
    """Check that we can pass the `data_source` argument to the precision-recall
    curve plot.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall(data_source="train")
    display.plot()
    ax = display.ax_
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "AP=1.00" in legend_texts

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    ax = display.ax_
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "AP=1.00" in legend_texts


@pytest.mark.parametrize(
    "fixture_name",
    [
        "logistic_binary_classification_with_train_test",
        "logistic_multiclass_classification_with_train_test",
    ],
)
def test_wrong_kwargs(pyplot, fixture_name, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `relplot_kwargs` argument.
    """
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    err_msg = "Line2D.set() got an unexpected keyword argument 'invalid'"
    with pytest.raises(AttributeError, match=re.escape(err_msg)):
        display.plot(relplot_kwargs={"invalid": "value"})


@pytest.mark.parametrize(
    "fixture_name",
    [
        "logistic_binary_classification_with_train_test",
        "logistic_multiclass_classification_with_train_test",
    ],
)
def test_relplot_kwargs(pyplot, fixture_name, request):
    """Check that we can pass keyword arguments to the precision-recall curve plot."""
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()

    display.plot()
    default_color = display.lines_[0].get_color()
    assert default_color == sns.color_palette()[:1][0]

    display.plot(relplot_kwargs={"color": "red"})
    assert display.lines_[0].get_color() == "red"
    assert mpl.colors.to_rgb(display.lines_[0].get_color()) != default_color

    display.set_style(relplot_kwargs={"color": "blue"}, policy="update")
    display.plot()
    assert display.lines_[0].get_color() == "blue"
    assert mpl.colors.to_rgb(display.lines_[0].get_color()) != default_color


def test_binary_classification_data_source(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check that we can pass the `data_source` argument to the precision-recall curve
    plot."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall(data_source="train")
    display.plot()
    assert display.ax_.get_legend().get_texts()[0].get_text() == "AP=1.00"
    assert "Data source: Train set" in display.figure_.get_suptitle()

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    assert display.ax_.get_legend().get_texts()[0].get_text() == "AP=1.00"
    assert "Data source: external set" in display.figure_.get_suptitle()


def test_multiclass_classification_data_source(
    pyplot, logistic_multiclass_classification_with_train_test
):
    """Check that we can pass the `data_source` argument to the precision-recall curve
    plot."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall(data_source="train")
    display.plot()
    for class_label in estimator.classes_:
        plot_data = display.frame(with_average_precision=True)
        average_precision = plot_data.query(f"label == {class_label}")[
            "average_precision"
        ].iloc[0]
        legend = display.ax_[class_label].get_legend()
        assert legend.get_texts()[0].get_text() == f"AP={average_precision:.2f}"
    assert (
        display.figure_.get_suptitle()
        == f"Precision-Recall Curve for {estimator.__class__.__name__}"
        f"\nData source: Train set"
    )

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    for class_label in estimator.classes_:
        plot_data = display.frame(with_average_precision=True)
        average_precision = plot_data.query(f"label == {class_label}")[
            "average_precision"
        ].iloc[0]
        assert legend.get_texts()[0].get_text() == f"AP={average_precision:.2f}"
    assert (
        display.figure_.get_suptitle()
        == f"Precision-Recall Curve for {estimator.__class__.__name__}"
        f"\nData source: external set"
    )


def test_binary_classification_data_source_both(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check the behavior of the precision-recall curve plot with binary data
    when data_source='both'.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall(data_source="both")
    display.plot()

    assert len(display.lines_) == 2
    plot_data = display.frame(with_average_precision=True)
    average_precision_train = plot_data.query("data_source == 'train'")[
        "average_precision"
    ].iloc[0]
    average_precision_test = plot_data.query("data_source == 'test'")[
        "average_precision"
    ].iloc[0]
    legend = display.ax_.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert legend_texts[0] == f"Train set (AP={average_precision_train:.2f})"
    assert legend_texts[1] == f"Test set (AP={average_precision_test:.2f})"


def test_multiclass_classification_data_source_both(
    pyplot, logistic_multiclass_classification_with_train_test
):
    """Check the behavior of the precision-recall curve plot with multiclass data
    when data_source='both'.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall(data_source="both")
    display.plot()

    n_classes = len(estimator.classes_)
    assert len(display.lines_) == n_classes * 2
    assert len(display.ax_) == n_classes

    for class_label in estimator.classes_:
        plot_data = display.frame(with_average_precision=True)
        average_precision_train = plot_data.query(
            f"label == {class_label} & data_source == 'train'"
        )["average_precision"].iloc[0]
        average_precision_test = plot_data.query(
            f"label == {class_label} & data_source == 'test'"
        )["average_precision"].iloc[0]
        legend = display.ax_[class_label].get_legend()
        legend_texts = [text.get_text() for text in legend.get_texts()]
        assert len(legend_texts) == 2
        assert legend_texts[0] == f"Train set (AP={average_precision_train:.2f})"
        assert legend_texts[1] == f"Test set (AP={average_precision_test:.2f})"


@pytest.mark.parametrize("with_average_precision", [False, True])
def test_frame_binary_classification(
    logistic_binary_classification_with_train_test, with_average_precision
):
    """Test the frame method with binary classification data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    df = report.metrics.precision_recall().frame(
        with_average_precision=with_average_precision
    )
    expected_index = []
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)

    if with_average_precision:
        assert df["average_precision"].nunique() == 1


@pytest.mark.parametrize("with_average_precision", [False, True])
def test_frame_multiclass_classification(
    logistic_multiclass_classification_with_train_test, with_average_precision
):
    """Test the frame method with multiclass classification data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    df = report.metrics.precision_recall().frame(
        with_average_precision=with_average_precision
    )
    expected_index = ["label"]
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["label"].nunique() == len(estimator.classes_)

    if with_average_precision:
        for (_), group in df.groupby(["label"], observed=True):
            assert group["average_precision"].nunique() == 1


@pytest.mark.parametrize("with_average_precision", [False, True])
def test_frame_multiclass_classification_data_source_both(
    logistic_multiclass_classification_with_train_test, with_average_precision
):
    """
    Test the frame method with multiclass classification data and data_source='both'.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    df = report.metrics.precision_recall(data_source="both").frame(
        with_average_precision=with_average_precision
    )
    expected_index = ["data_source", "label"]
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["label"].nunique() == len(estimator.classes_)
    assert set(df["data_source"].unique()) == {"train", "test"}

    if with_average_precision:
        for _, group in df.groupby(["label"], observed=True):
            assert group["average_precision"].nunique() == 2


def test_legend(
    pyplot,
    logistic_binary_classification_with_train_test,
    logistic_multiclass_classification_with_train_test,
):
    """Check the rendering of the legend for with an `EstimatorReport`."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="upper center", position="inside")

    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_[0], loc="upper center", position="inside")


def test_binary_classification_constructor(
    logistic_binary_classification_with_train_test,
):
    """Check that the dataframe has the correct structure at initialization."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()

    index_columns = ["estimator", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator"].unique()[0] == report.estimator_name_
        assert df["split"].isnull().all()
        assert df["label"].unique() == 1

    assert len(display.average_precision) == 1


def test_multiclass_classification_constructor(
    logistic_multiclass_classification_with_train_test,
):
    """Check that the dataframe has the correct structure at initialization."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()

    index_columns = ["estimator", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator"].unique()[0] == report.estimator_name_
        assert df["split"].isnull().all()
        np.testing.assert_array_equal(df["label"].unique(), estimator.classes_)

    assert len(display.average_precision) == len(estimator.classes_)


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        ("logistic_binary_classification_with_train_test", ["None", "auto"]),
        ("logistic_multiclass_classification_with_train_test", ["auto", "label"]),
    ],
)
def test_invalid_subplot_by(fixture_name, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    valid_values_str = ", ".join(valid_values)
    err_msg = f"subplot_by must be one of {valid_values_str}. Got 'invalid' instead."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "fixture_name, subplot_by, expected_len",
    [
        ("logistic_binary_classification_with_train_test", None, 0),
        ("logistic_multiclass_classification_with_train_test", "label", 3),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by, expected_len, request):
    """Check that we can pass `None` to `subplot_by`."""
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    display.plot(subplot_by=subplot_by)
    if subplot_by is None:
        assert isinstance(display.ax_, mpl.axes.Axes)
    else:
        assert len(display.ax_) == expected_len
