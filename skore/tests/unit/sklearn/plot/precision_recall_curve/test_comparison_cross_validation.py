from itertools import product

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from skore import ComparisonReport, CrossValidationReport
from skore.sklearn._plot.metrics.precision_recall_curve import (
    PrecisionRecallCurveDisplay,
)
from skore.sklearn._plot.utils import sample_mpl_colormap
from skore.utils._testing import check_legend_position, check_precision_recall_frame
from skore.utils._testing import (
    check_precision_recall_curve_display_data as check_display_data,
)


@pytest.fixture
def binary_classification_report(binary_classification_data_no_split):
    _, X, y = binary_classification_data_no_split
    estimator_1 = LogisticRegression()
    estimator_2 = LogisticRegression(C=10)
    report = ComparisonReport(
        reports={
            "estimator_1": CrossValidationReport(estimator_1, X, y),
            "estimator_2": CrossValidationReport(estimator_2, X, y),
        }
    )
    return report


@pytest.fixture
def multiclass_classification_report(multiclass_classification_data_no_split):
    _, X, y = multiclass_classification_data_no_split
    estimator_1 = LogisticRegression()
    estimator_2 = LogisticRegression(C=10)
    report = ComparisonReport(
        reports={
            "estimator_1": CrossValidationReport(estimator_1, X, y),
            "estimator_2": CrossValidationReport(estimator_2, X, y),
        }
    )
    return report


def test_binary_classification(pyplot, binary_classification_report):
    """
    Check the behaviour of `precision_recall` when ML task is "binary-classification".
    """
    report = binary_classification_report
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    pos_label = 1
    n_reports = len(report.reports_)
    n_splits = len(report.reports_[0].estimator_reports_)

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == n_reports * n_splits
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for i, estimator_name in enumerate(report.report_names_):
        precision_recall_mpl = display.lines_[i * n_splits]
        assert isinstance(precision_recall_mpl, Line2D)

        average_precision = display.average_precision.query(
            f"label == {pos_label} & estimator_name == '{estimator_name}'"
        )["average_precision"]

        assert precision_recall_mpl.get_label() == (
            f"{estimator_name} (AUC = {average_precision.mean():0.2f} "
            f"+/- {average_precision.std():0.2f})"
        )
        assert list(precision_recall_mpl.get_color()[:3]) == list(default_colors[i][:3])

    assert isinstance(display.ax_, mpl.axes.Axes)
    check_legend_position(display.ax_, loc="lower left", position="inside")
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "Test set"
    assert len(legend.get_texts()) == n_reports

    assert display.ax_.get_xlabel() == "Recall\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "Precision\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)
    assert display.ax_.get_title() == "Precision-Recall Curve"


def test_multiclass_classification(pyplot, multiclass_classification_report):
    """
    Check the behaviour of `precision_recall` when ML task is
    "multiclass-classification" and `pos_label` is None.
    """
    report = multiclass_classification_report
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    labels = display.precision_recall["label"].cat.categories
    n_reports = len(report.reports_)
    n_splits = len(report.reports_[0].estimator_reports_)

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == n_reports * len(labels) * n_splits

    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for i, ((estimator_idx, estimator_name), label) in enumerate(
        product(enumerate(report.report_names_), labels)
    ):
        precision_recall_mpl = display.lines_[i * n_splits]
        assert isinstance(precision_recall_mpl, Line2D)

        average_precision = display.average_precision.query(
            f"label == {label} & estimator_name == '{estimator_name}'"
        )["average_precision"]

        assert precision_recall_mpl.get_label() == (
            f"{estimator_name} (AUC = {average_precision.mean():0.2f} "
            f"+/- {average_precision.std():0.2f})"
        )
        assert list(precision_recall_mpl.get_color()[:3]) == list(
            default_colors[estimator_idx][:3]
        )

    assert isinstance(display.ax_, np.ndarray)
    for label, ax in zip(labels, display.ax_, strict=False):
        check_legend_position(ax, loc="lower left", position="inside")
        legend = ax.get_legend()
        assert legend.get_title().get_text() == "Test set"
        assert len(legend.get_texts()) == n_reports

        assert ax.get_xlabel() == f"Recall\n(Positive label: {label})"
        assert ax.get_ylabel() == f"Precision\n(Positive label: {label})"
        assert ax.get_adjustable() == "box"
        assert ax.get_aspect() in ("equal", 1.0)
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    assert display.figure_.get_suptitle() == "Precision-Recall Curve"


def test_binary_classification_wrong_kwargs(pyplot, binary_classification_report):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `pr_curve_kwargs` argument."""
    report = binary_classification_report
    display = report.metrics.precision_recall()
    err_msg = (
        "You intend to plot multiple curves. We expect `pr_curve_kwargs` to be a "
        "list of dictionaries with the same length as the number of curves. "
        "Got 2 instead of 10."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs=[{}, {}])


@pytest.mark.parametrize("pr_curve_kwargs", [[{"color": "red"}] * 10])
def test_binary_classification_kwargs(
    pyplot, binary_classification_report, pr_curve_kwargs
):
    """Check that we can pass keyword arguments to the PR curve plot."""
    report = binary_classification_report
    display = report.metrics.precision_recall()
    display.plot(pr_curve_kwargs=pr_curve_kwargs)
    assert display.lines_[0].get_color() == "red"

    # check the `.style` display setter
    display.plot()  # default style
    assert display.lines_[0].get_color() == (
        np.float64(0.12156862745098039),
        np.float64(0.4666666666666667),
        np.float64(0.7058823529411765),
        np.float64(1.0),
    )

    display.set_style(pr_curve_kwargs=pr_curve_kwargs)
    display.plot()
    assert display.lines_[0].get_color() == "red"

    # overwrite the style that was set above
    display.plot(pr_curve_kwargs=[{"color": "#1f77b4"}] * 10)
    assert display.lines_[0].get_color() == "#1f77b4"


def test_multiclass_classification_wrong_kwargs(
    pyplot, multiclass_classification_report
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `pr_curve_kwargs` argument."""
    report = multiclass_classification_report
    display = report.metrics.precision_recall()
    err_msg = "You intend to plot multiple curves."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs=[{}, {}])

    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs={})


def test_multiclass_classification_kwargs(pyplot, multiclass_classification_report):
    """Check that we can pass keyword arguments to the PR curve plot for
    multiclass classification."""
    report = multiclass_classification_report
    display = report.metrics.precision_recall()
    display.plot(
        pr_curve_kwargs=(
            [{"color": "red"}] * 10
            + [{"color": "blue"}] * 10
            + [{"color": "green"}] * 10
        )
    )
    assert display.lines_[0].get_color() == "red"
    assert display.lines_[10].get_color() == "blue"
    assert display.lines_[20].get_color() == "green"

    display.plot()

    display.plot(despine=False)
    assert display.ax_[0].spines["top"].get_visible()


def test_frame_binary_classification(binary_classification_report):
    """Test the frame method with binary classification comparison cross-validation
    data."""
    report = binary_classification_report
    display = report.metrics.precision_recall()
    df = display.frame()

    check_precision_recall_frame(
        df,
        expected_n_splits=report.reports_[0]._cv_splitter.n_splits,
        multiclass=False,
    )

    assert df["estimator_name"].nunique() == len(report.reports_)


def test_frame_multiclass_classification(multiclass_classification_report):
    """Test the frame method with multiclass classification comparison cross-validation
    data."""
    report = multiclass_classification_report
    display = report.metrics.precision_recall()
    df = display.frame()

    check_precision_recall_frame(
        df,
        expected_n_splits=report.reports_[0]._cv_splitter.n_splits,
        multiclass=True,
    )

    assert df["estimator_name"].nunique() == len(report.reports_)
    assert df["method"].unique() == ["OvR"]
