"""Test the `roc_curve` display method."""

from itertools import product

import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from skore import ComparisonReport, CrossValidationReport
from skore.sklearn._plot.metrics.roc_curve import RocCurveDisplay
from skore.sklearn._plot.utils import sample_mpl_colormap


def test_binary_classification(pyplot, binary_classification_data_no_split):
    """Check the behaviour of `roc_curve` when ML task is "binary-classification"."""
    _, X, y = binary_classification_data_no_split
    estimator_1 = LogisticRegression()
    estimator_2 = LogisticRegression(C=10)
    report = ComparisonReport(
        reports={
            "estimator_1": CrossValidationReport(estimator_1, X, y),
            "estimator_2": CrossValidationReport(estimator_2, X, y),
        }
    )
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)

    pos_label = 1
    n_reports = len(report.reports_)
    n_splits = report.reports_[0]._cv_splitter.n_splits

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == n_reports * n_splits
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for i, estimator_name in enumerate(report.report_names_):
        roc_curve_mpl = display.lines_[i * n_splits]
        assert isinstance(roc_curve_mpl, Line2D)
        auc = display.roc_auc.query(
            f"label == {pos_label} & estimator_name == '{estimator_name}'"
        )["roc_auc"]

        assert roc_curve_mpl.get_label() == (
            f"{report.report_names_[i]} (AUC = {auc.mean():0.2f} +/- {auc.std():0.2f})"
        )
        assert list(roc_curve_mpl.get_color()[:3]) == list(default_colors[i][:3])

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == r"Binary-Classification on $\bf{test}$ set"
    assert len(legend.get_texts()) == n_reports + 1

    assert display.ax_.get_xlabel() == "False Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "True Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_multiclass_classification(pyplot, multiclass_classification_data_no_split):
    """Check the behaviour of `roc_curve` when ML task is "multiclass-classification"
    and `pos_label` is None."""
    _, X, y = multiclass_classification_data_no_split
    estimator_1 = LogisticRegression()
    estimator_2 = LogisticRegression(C=10)
    report = ComparisonReport(
        reports={
            "estimator_1": CrossValidationReport(estimator_1, X, y),
            "estimator_2": CrossValidationReport(estimator_2, X, y),
        }
    )
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)

    labels = display.roc_curve["label"].unique()
    n_reports = len(report.reports_)
    n_splits = report.reports_[0]._cv_splitter.n_splits

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == n_reports * len(labels) * n_splits
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for i, ((estimator_idx, estimator_name), label) in enumerate(
        product(enumerate(report.report_names_), labels)
    ):
        roc_curve_mpl = display.lines_[i * n_splits]
        assert isinstance(roc_curve_mpl, Line2D)

        auc = display.roc_auc.query(
            f"label == {label} & estimator_name == '{estimator_name}'"
        )["roc_auc"]

        assert roc_curve_mpl.get_label() == (
            f"{estimator_name} (AUC = {auc.mean():0.2f} +/- {auc.std():0.2f})"
        )
        assert list(roc_curve_mpl.get_color()[:3]) == list(
            default_colors[estimator_idx][:3]
        )

    assert isinstance(display.chance_level_, list)
    assert isinstance(display.chance_level_[0], mpl.lines.Line2D)
    assert display.chance_level_[0].get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_[0].get_color() == "k"

    assert isinstance(display.ax_, np.ndarray)
    for label, ax in zip(labels, display.ax_):
        legend = ax.get_legend()
        assert (
            legend.get_title().get_text()
            == r"Multiclass-Classification on $\bf{test}$ set"
        )
        assert len(legend.get_texts()) == n_reports + 1

        assert ax.get_xlabel() == f"False Positive Rate\n(Positive label: {label})"
        assert ax.get_ylabel() == f"True Positive Rate\n(Positive label: {label})"
        assert ax.get_adjustable() == "box"
        assert ax.get_aspect() in ("equal", 1.0)
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
