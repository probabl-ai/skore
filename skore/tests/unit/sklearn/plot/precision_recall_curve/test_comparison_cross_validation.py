from itertools import product

import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from skore import ComparisonReport, CrossValidationReport
from skore.sklearn._plot.metrics.precision_recall_curve import (
    PrecisionRecallCurveDisplay,
)
from skore.sklearn._plot.utils import sample_mpl_colormap

from .utils import check_display_data


def test_binary_classification(pyplot, binary_classification_data_no_split):
    """
    Check the behaviour of `precision_recall` when ML task is "binary-classification".
    """
    X, y = binary_classification_data_no_split
    estimator_1 = LogisticRegression()
    estimator_2 = LogisticRegression(C=10)
    report = ComparisonReport(
        reports={
            "estimator_1": CrossValidationReport(estimator_1, X, y),
            "estimator_2": CrossValidationReport(estimator_2, X, y),
        }
    )
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
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == r"Binary-Classification on $\bf{test}$ set"
    assert len(legend.get_texts()) == n_reports

    assert display.ax_.get_xlabel() == "Recall\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "Precision\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_multiclass_classification(pyplot, multiclass_classification_data_no_split):
    """
    Check the behaviour of `precision_recall` when ML task is
    "multiclass-classification" and `pos_label` is None.
    """
    X, y = multiclass_classification_data_no_split
    estimator_1 = LogisticRegression()
    estimator_2 = LogisticRegression(C=10)
    report = ComparisonReport(
        reports={
            "estimator_1": CrossValidationReport(estimator_1, X, y),
            "estimator_2": CrossValidationReport(estimator_2, X, y),
        }
    )
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    labels = display.precision_recall["label"].unique()
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
    for label, ax in zip(labels, display.ax_):
        legend = ax.get_legend()
        assert (
            legend.get_title().get_text()
            == r"Multiclass-Classification on $\bf{test}$ set"
        )
        assert len(legend.get_texts()) == n_reports

        assert ax.get_xlabel() == f"Recall\n(Positive label: {label})"
        assert ax.get_ylabel() == f"Precision\n(Positive label: {label})"
        assert ax.get_adjustable() == "box"
        assert ax.get_aspect() in ("equal", 1.0)
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
