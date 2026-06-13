"""Test the `roc_curve` display method."""

import matplotlib as mpl
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from skore import ComparisonReport, CrossValidationReport, compare, evaluate


def test_data_source_both_legend_matches_curves(pyplot):
    """Legend colors must match the drawn curves with ``data_source="both"``.

    Non-regression test for https://github.com/probabl-ai/skore/issues/2925: with
    categorical estimator/data_source columns (e.g. on pandas >= 3), seaborn draws in
    category order while the legend was built in order of appearance, so the legend
    color for an estimator disagreed with that estimator's actual curves. The
    estimators are chosen so the order they are passed
    (``HistGradientBoostingClassifier`` then ``DecisionTreeClassifier``) differs from
    alphabetical order, and so that one clearly outperforms the other (distinct AUC).
    """
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = load_breast_cancer(return_X_y=True)
    report = compare(
        [
            evaluate(HistGradientBoostingClassifier(), X, y, splitter=2),
            evaluate(DecisionTreeClassifier(max_depth=1), X, y, splitter=2),
        ]
    )
    display = report.metrics.roc(data_source="both")
    fig = display.plot(subplot_by=None, label=0)
    ax = fig.axes[0]

    tpr_by_color: dict = {}
    for line in ax.get_lines():
        if not line.get_label().startswith("_child"):
            continue
        color = tuple(line.get_color())
        x, y_vals = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        low_fpr = x <= 0.1
        if low_fpr.any():
            tpr_by_color.setdefault(color, []).append(y_vals[low_fpr].max())
    curve_colors = {
        c: np.mean(v) for c, v in tpr_by_color.items() if c[:3] != (0, 0, 0)
    }
    strong_curve_color = max(curve_colors, key=curve_colors.get)

    legend = ax.get_legend()
    hgbc_legend_color = next(
        tuple(handle.get_color())
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
        if text.get_text().startswith("HistGradientBoostingClassifier")
    )
    assert hgbc_legend_color[:3] == strong_curve_color[:3]

    entries = [
        (text.get_text(), handle)
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
        if "Chance level" not in text.get_text()
    ]
    names = [text.split(" - ")[0] for text, _ in entries]
    assert names[0] == "HistGradientBoostingClassifier"

    for text, handle in entries:
        assert handle.get_linestyle() == ("--" if "Train" in text else "-")


def test_legend_binary_classification(
    pyplot,
    comparison_cross_validation_reports_binary_classification,
    comparison_cross_validation_reports_binary_classification_figure_axes,
):
    """
    Check the legend of the ROC curve plot with binary data.
    """
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.roc()
    _, axes = comparison_cross_validation_reports_binary_classification_figure_axes
    for ax, estimator in zip(axes, report.reports_, strict=True):
        assert isinstance(ax, mpl.axes.Axes)
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        labels = display.roc_curve["label"].cat.categories
        for label_idx, label in enumerate(labels):
            plot_data = display.frame(with_roc_auc=True)
            roc_auc = plot_data.query(f"label == {label} & estimator == '{estimator}'")[
                "roc_auc"
            ]
            assert (
                legend_texts[label_idx] == f"{label} (AUC={roc_auc.mean():.2f}"
                f"±{roc_auc.std():.2f})"
            )
        assert len(legend_texts) == len(labels) + 1
        assert legend_texts[-1] == "Chance level (AUC = 0.5)"


def test_legend_multiclass_classification(
    pyplot,
    comparison_cross_validation_reports_multiclass_classification,
    comparison_cross_validation_reports_multiclass_classification_figure_axes,
):
    """
    Check the legend of the ROC curve plot with multiclass data.
    """
    report = comparison_cross_validation_reports_multiclass_classification
    display = report.metrics.roc()
    _, axes = comparison_cross_validation_reports_multiclass_classification_figure_axes

    labels = display.roc_curve["label"].cat.categories

    for ax, estimator in zip(axes, report.reports_, strict=True):
        assert isinstance(ax, mpl.axes.Axes)
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        for label_idx, label in enumerate(labels):
            plot_data = display.frame(with_roc_auc=True)
            roc_auc = plot_data.query(f"label == {label} & estimator == '{estimator}'")[
                "roc_auc"
            ]
            assert (
                legend_texts[label_idx] == f"{label} (AUC={roc_auc.mean():.2f}"
                f"±{roc_auc.std():.2f})"
            )
        assert len(legend_texts) == len(labels) + 1
        assert legend_texts[-1] == "Chance level (AUC = 0.5)"


def test_multiclass_str_labels_roc_plot(pyplot):
    """Regression test for issue #2183 with multiclass comparison reports.

    Using string labels backed by numpy.str_ should not break
    `roc().plot()` for a multiclass ComparisonReport.
    """
    iris = load_iris(as_frame=True)
    X, y_int = iris.data, iris.target
    y = iris.target_names[y_int]

    report_1 = CrossValidationReport(LogisticRegression(max_iter=500), X=X, y=y)
    report_2 = CrossValidationReport(LogisticRegression(max_iter=1000), X=X, y=y)
    report = ComparisonReport([report_1, report_2])
    report.metrics.roc().plot()


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        (
            "comparison_cross_validation_reports_binary_classification",
            ["auto", "estimator", "label"],
        ),
        (
            "comparison_cross_validation_reports_multiclass_classification",
            ["auto", "estimator", "label"],
        ),
    ],
)
def test_invalid_subplot_by(fixture_name, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    report = request.getfixturevalue(fixture_name)
    display = report.metrics.roc()
    err_msg = (
        f"subplot_by must be one of {', '.join(valid_values)}. Got 'invalid' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "fixture_name, subplot_by_tuples",
    [
        (
            "comparison_cross_validation_reports_binary_classification",
            [("label", 2), ("estimator", 2)],
        ),
        (
            "comparison_cross_validation_reports_multiclass_classification",
            [("label", 3), ("estimator", 2)],
        ),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by_tuples, request):
    """Check that we can pass non default values to `subplot_by`."""
    report = request.getfixturevalue(fixture_name)
    display = report.metrics.roc()
    for subplot_by, expected_len in subplot_by_tuples:
        fig = display.plot(subplot_by=subplot_by)
        axes = fig.axes
        if subplot_by is None:
            assert len(axes) == 1
            assert isinstance(axes[0], mpl.axes.Axes)
        else:
            assert len(axes) == expected_len
