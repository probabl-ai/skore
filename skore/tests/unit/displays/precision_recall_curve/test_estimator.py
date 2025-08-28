import matplotlib as mpl
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore._sklearn._plot import PrecisionRecallCurveDisplay
from skore._sklearn._plot.utils import sample_mpl_colormap
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
    average_precision = display.average_precision.query(
        f"label == {estimator.classes_[1]}"
    )["average_precision"].item()
    assert (
        precision_recall_curve_mpl.get_label()
        == f"Test set (AP = {average_precision:0.2f})"
    )
    assert precision_recall_curve_mpl.get_color() == "#1f77b4"  # tab:blue in hex

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == ""
    assert len(legend.get_texts()) == 1

    assert display.ax_.get_xlabel() == "Recall\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "Precision\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)
    assert (
        display.ax_.get_title()
        == f"Precision-Recall Curve for {estimator.__class__.__name__}"
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
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for class_label, expected_color in zip(
        estimator.classes_, default_colors, strict=False
    ):
        precision_recall_curve_mpl = display.lines_[class_label]
        assert isinstance(precision_recall_curve_mpl, mpl.lines.Line2D)
        average_precision = display.average_precision.query(f"label == {class_label}")[
            "average_precision"
        ].item()
        assert precision_recall_curve_mpl.get_label() == (
            f"{str(class_label).title()} (AP = {average_precision:0.2f})"
        )
        assert precision_recall_curve_mpl.get_color() == expected_color

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "Test set"
    assert len(legend.get_texts()) == 3

    assert display.ax_.get_xlabel() == "Recall"
    assert display.ax_.get_ylabel() == "Precision"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)
    assert (
        display.ax_.get_title()
        == f"Precision-Recall Curve for {estimator.__class__.__name__}"
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
    assert display.lines_[0].get_label() == "Train set (AP = 1.00)"

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    assert display.lines_[0].get_label() == "AP = 1.00"


def test_pr_curve_kwargs(
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
    for pr_curve_kwargs in ({"color": "red"}, [{"color": "red"}]):
        display = report.metrics.precision_recall()
        display.plot(pr_curve_kwargs=pr_curve_kwargs)
        assert display.lines_[0].get_color() == "red"

        # check the `.style` display setter
        display.plot()  # default style
        assert display.lines_[0].get_color() == "#1f77b4"
        display.set_style(pr_curve_kwargs=pr_curve_kwargs)
        display.plot()
        assert display.lines_[0].get_color() == "red"
        display.plot(pr_curve_kwargs=pr_curve_kwargs)
        assert display.lines_[0].get_color() == "red"

        # reset to default style since next call to `precision_recall` will use the
        # cache
        display.set_style(pr_curve_kwargs={"color": "#1f77b4"})

    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    display.plot(
        pr_curve_kwargs=[dict(color="red"), dict(color="blue"), dict(color="green")],
    )
    assert display.lines_[0].get_color() == "red"
    assert display.lines_[1].get_color() == "blue"
    assert display.lines_[2].get_color() == "green"

    display.plot(despine=False)
    assert display.ax_.spines["top"].get_visible()
    assert display.ax_.spines["right"].get_visible()


def test_wrong_kwargs(
    pyplot,
    logistic_binary_classification_with_train_test,
    logistic_multiclass_classification_with_train_test,
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `roc_curve_kwargs` argument.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    err_msg = (
        "You intend to plot a single curve. We expect `pr_curve_kwargs` to be a "
        "dictionary."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs=[{}, {}])

    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    err_msg = (
        "You intend to plot multiple curves. We expect `pr_curve_kwargs` to be a "
        "list of dictionaries."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs=[{}, {}])

    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs={})


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
    assert display.lines_[0].get_label() == "Train set (AP = 1.00)"

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    assert display.lines_[0].get_label() == "AP = 1.00"


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
        average_precision = display.average_precision.query(f"label == {class_label}")[
            "average_precision"
        ].item()
        assert display.lines_[class_label].get_label() == (
            f"{str(class_label).title()} (AP = {average_precision:0.2f})"
        )

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    for class_label in estimator.classes_:
        average_precision = display.average_precision.query(f"label == {class_label}")[
            "average_precision"
        ].item()
        assert display.lines_[class_label].get_label() == (
            f"{str(class_label).title()} (AP = {average_precision:0.2f})"
        )


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


def test_legend(
    pyplot,
    logistic_binary_classification_with_train_test,
    logistic_multiclass_classification_with_train_test,
):
    """Check the rendering of the legend for with an `EstimatorReport`."""

    # binary classification
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="lower left", position="inside")

    # multiclass classification <= 5 classes
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="lower left", position="inside")

    # multiclass classification > 5 classes
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
    check_legend_position(display.ax_, loc="upper left", position="outside")


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

    index_columns = ["estimator_name", "split_index", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator_name"].unique() == report.estimator_name_
        assert df["split_index"].isnull().all()
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

    index_columns = ["estimator_name", "split_index", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator_name"].unique() == report.estimator_name_
        assert df["split_index"].isnull().all()
        np.testing.assert_array_equal(df["label"].unique(), estimator.classes_)

    assert len(display.average_precision) == len(estimator.classes_)
