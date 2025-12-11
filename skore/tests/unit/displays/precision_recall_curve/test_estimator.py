import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pytest
from numpy.testing import assert_allclose
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

    assert isinstance(display.ax_, np.ndarray)
    ax = display.ax_[0]
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    average_precision = display.average_precision.query(
        f"label == {estimator.classes_[1]}"
    )["average_precision"].item()
    assert f"AP={average_precision:.2f}" in legend_texts

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

    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(estimator.classes_)

    for idx, class_label in enumerate(estimator.classes_):
        ax = display.ax_[idx]
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]
        average_precision = display.average_precision.query(f"label == {class_label}")[
            "average_precision"
        ].item()
        assert f"AP={average_precision:.2f}" in legend_texts

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
    ax = display.ax_[0]
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "AP=1.00" in legend_texts

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    ax = display.ax_[0]
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "AP=1.00" in legend_texts


def test_relplot_kwargs(
    pyplot,
    logistic_binary_classification_with_train_test,
    logistic_multiclass_classification_with_train_test,
):
    """Check that we can pass keyword arguments to the precision-recall curve plot."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()

    display.plot()
    default_linewidth = display.lines_[0].get_linewidth()

    display.plot(relplot_kwargs={"linewidth": 2})
    assert len(display.lines_) == 1
    assert display.lines_[0].get_linewidth() == 2
    assert display.lines_[0].get_linewidth() != default_linewidth

    display.plot()
    display.set_style(relplot_kwargs={"linewidth": 2}, policy="update")
    display.plot()
    assert len(display.lines_) == 1
    assert display.lines_[0].get_linewidth() == 2

    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()

    display.plot(relplot_kwargs={"alpha": 0.5})
    assert len(display.lines_) == len(estimator.classes_)
    for line in display.lines_:
        assert line.get_alpha() == 0.5

    display.plot(despine=False)
    for ax in display.ax_:
        assert ax is not None

    display = report.metrics.precision_recall(data_source="both")
    display.plot(relplot_kwargs={"palette": ["red", "blue"]})
    assert len(display.lines_) == len(estimator.classes_) * 2

    expected_colors = ["red", "blue"]
    first_subplot_lines = display.ax_[0].get_lines()
    for line, expected_color in zip(first_subplot_lines, expected_colors, strict=True):
        line_color = line.get_color()
        expected_rgb = mcolors.to_rgb(expected_color)
        actual_rgb = mcolors.to_rgb(line_color)
        assert_allclose(expected_rgb, actual_rgb, atol=0.01)


def test_wrong_kwargs(
    pyplot,
    logistic_binary_classification_with_train_test,
    logistic_multiclass_classification_with_train_test,
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `relplot_kwargs` argument.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    with pytest.raises(ValueError, match="subplot_by"):
        display.plot(subplot_by="invalid")

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        display.plot(non_existent_kwarg="value")


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
    ax = display.ax_[0]
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    # When there's only one data source, legend shows just AP, not data source
    assert "AP=1.00" in legend_texts

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    ax = display.ax_[0]
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "AP=1.00" in legend_texts


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
    for idx, class_label in enumerate(estimator.classes_):
        ax = display.ax_[idx]
        legend = ax.get_legend()
        legend_texts = [text.get_text() for text in legend.get_texts()]
        average_precision = display.average_precision.query(f"label == {class_label}")[
            "average_precision"
        ].item()
        assert f"AP={average_precision:.2f}" in legend_texts

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    for idx, class_label in enumerate(estimator.classes_):
        ax = display.ax_[idx]
        legend = ax.get_legend()
        legend_texts = [text.get_text() for text in legend.get_texts()]
        average_precision = display.average_precision.query(f"label == {class_label}")[
            "average_precision"
        ].item()
        assert f"AP={average_precision:.2f}" in legend_texts


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

    ax = display.ax_[0]
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert any("Train set (AP=" in text for text in legend_texts)
    assert any("Test set (AP=" in text for text in legend_texts)
    assert len(legend_texts) == 2


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

    for idx in range(len(estimator.classes_)):
        ax = display.ax_[idx]
        legend = ax.get_legend()
        legend_texts = [text.get_text() for text in legend.get_texts()]
        assert len(legend_texts) == 2
        assert any("Train set (AP=" in text for text in legend_texts)
        assert any("Test set (AP=" in text for text in legend_texts)


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
    check_legend_position(display.ax_[0], loc="upper center", position="inside")

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

    index_columns = ["estimator_name", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator_name"].unique() == report.estimator_name_
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

    index_columns = ["estimator_name", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator_name"].unique() == report.estimator_name_
        assert df["split"].isnull().all()
        np.testing.assert_array_equal(df["label"].unique(), estimator.classes_)

    assert len(display.average_precision) == len(estimator.classes_)
