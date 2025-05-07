"""Test the `roc_curve` display method."""

import matplotlib as mpl
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from skore import ComparisonReport, CrossValidationReport
from skore.sklearn._plot.metrics.roc_curve import RocCurveDisplay
from skore.sklearn._plot.utils import sample_mpl_colormap


def test_binary_classification(pyplot):
    """Check the behaviour of `roc_curve` when ML task is "binary-classification"."""
    X, y = make_classification(class_sep=0.1, random_state=42)
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

    # check the structure of the attributes
    pos_label = 1
    n_reports = len(report.reports_)
    for attr_name in ("fpr", "tpr", "roc_auc"):
        assert isinstance(getattr(display, attr_name), list)
        assert len(getattr(display, attr_name)) == n_reports

        attr = getattr(display, attr_name)
        for i, cv_report in enumerate(report.reports_):
            # Positive class is 1
            assert isinstance(attr[i][pos_label], list)
            assert len(attr[i][pos_label]) == cv_report._cv_splitter.n_splits

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == n_reports
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for i in range(n_reports):
        roc_curve_mpl = display.lines_[i]
        assert isinstance(roc_curve_mpl, mpl.collections.LineCollection)
        mean_auc = np.mean(display.roc_auc[i][pos_label])
        std_auc = np.std(display.roc_auc[i][pos_label], mean=mean_auc)
        assert roc_curve_mpl.get_label() == (
            f"{report.report_names_[i]} (AUC = {mean_auc:0.2f} +/- {std_auc:0.2f})"
        )
        assert list(roc_curve_mpl.get_color()[0][:3]) == list(default_colors[i][:3])

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


def test_multiclass(pyplot):
    """Check the behaviour of `roc_curve` when ML task is "multiclass-classification"
    and `pos_label` is None."""
    X, y = make_classification(
        class_sep=0.1, n_classes=3, n_clusters_per_class=1, random_state=42
    )
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

    # check the structure of the attributes
    labels = display.fpr[0].keys()
    n_reports = len(report.reports_)
    for attr_name in ("fpr", "tpr", "roc_auc"):
        assert isinstance(getattr(display, attr_name), list)
        assert len(getattr(display, attr_name)) == n_reports

        attr = getattr(display, attr_name)
        for i, cv_report in enumerate(report.reports_):
            for label in labels:
                assert isinstance(attr[i][label], list)
                assert len(attr[i][label]) == cv_report._cv_splitter.n_splits

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(labels)
    assert len(display.lines_[0]) == n_reports
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for report_idx, (label_idx, label) in zip(range(n_reports), enumerate(labels)):
        roc_curve_mpl = display.lines_[label_idx][report_idx]
        assert isinstance(roc_curve_mpl, mpl.collections.LineCollection)

        roc_auc = display.roc_auc[report_idx][label]
        mean_auc = np.mean(roc_auc)
        std_auc = np.std(roc_auc, mean=mean_auc)
        assert roc_curve_mpl.get_label() == (
            f"{report.report_names_[report_idx]} "
            f"(AUC = {mean_auc:0.2f} +/- {std_auc:0.2f})"
        )
        assert list(roc_curve_mpl.get_color()[0][:3]) == list(
            default_colors[report_idx][:3]
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
