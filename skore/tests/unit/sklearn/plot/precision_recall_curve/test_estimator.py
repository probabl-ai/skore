import matplotlib as mpl
import pytest
from skore import EstimatorReport
from skore.sklearn._plot import PrecisionRecallCurveDisplay
from skore.sklearn._plot.utils import sample_mpl_colormap

from .utils import check_display_data


def test_binary_classification(pyplot, binary_classification_data):
    """Check the attributes and default plotting behaviour of the
    precision-recall curve plot with binary data.
    """
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
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
    )["average_precision"].iloc[0]
    assert (
        precision_recall_curve_mpl.get_label()
        == f"Test set (AP = {average_precision:0.2f})"
    )
    assert precision_recall_curve_mpl.get_color() == "#1f77b4"  # tab:blue in hex

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 1

    assert display.ax_.get_xlabel() == "Recall\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "Precision\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_multiclass_classification(pyplot, multiclass_classification_data):
    """Check the attributes and default plotting behaviour of the precision-recall
    curve plot with multiclass data.
    """
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
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
    for class_label, expected_color in zip(estimator.classes_, default_colors):
        precision_recall_curve_mpl = display.lines_[class_label]
        assert isinstance(precision_recall_curve_mpl, mpl.lines.Line2D)
        average_precision = display.average_precision.query(f"label == {class_label}")[
            "average_precision"
        ].iloc[0]
        assert precision_recall_curve_mpl.get_label() == (
            f"{str(class_label).title()} - test set (AP = {average_precision:0.2f})"
        )
        assert precision_recall_curve_mpl.get_color() == expected_color

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 3

    assert display.ax_.get_xlabel() == "Recall"
    assert display.ax_.get_ylabel() == "Precision"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_data_source(pyplot, binary_classification_data):
    """Check that we can pass the `data_source` argument to the precision-recall
    curve plot.
    """
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall(data_source="train")
    display.plot()
    assert display.lines_[0].get_label() == "Train set (AP = 0.93)"

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    assert display.lines_[0].get_label() == "AP = 0.93"


def test_pr_curve_kwargs(
    pyplot, binary_classification_data, multiclass_classification_data
):
    """Check that we can pass keyword arguments to the precision-recall curve plot."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
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

    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
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
    pyplot, binary_classification_data, multiclass_classification_data
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `roc_curve_kwargs` argument.
    """
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
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

    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
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


def test_binary_classification_data_source(pyplot, binary_classification_data):
    """Check that we can pass the `data_source` argument to the precision-recall curve
    plot."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall(data_source="train")
    display.plot()
    assert display.lines_[0].get_label() == "Train set (AP = 0.93)"

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    assert display.lines_[0].get_label() == "AP = 0.93"


def test_multiclass_classification_data_source(pyplot, multiclass_classification_data):
    """Check that we can pass the `data_source` argument to the precision-recall curve
    plot."""
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall(data_source="train")
    display.plot()
    for class_label in estimator.classes_:
        average_precision = display.average_precision.query(f"label == {class_label}")[
            "average_precision"
        ].iloc[0]
        assert display.lines_[class_label].get_label() == (
            f"{str(class_label).title()} - train set (AP = {average_precision:0.2f})"
        )

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    for class_label in estimator.classes_:
        average_precision = display.average_precision.query(f"label == {class_label}")[
            "average_precision"
        ].iloc[0]
        assert display.lines_[class_label].get_label() == (
            f"{str(class_label).title()} - AP = {average_precision:0.2f}"
        )
