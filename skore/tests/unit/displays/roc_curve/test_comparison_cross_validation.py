"""Test the `roc_curve` display method."""

import matplotlib as mpl
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from skore import ComparisonReport, CrossValidationReport


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

        plot_data = display.frame(with_roc_auc=True)
        roc_auc = plot_data.query(f"estimator == '{estimator}'")["roc_auc"]
        assert legend_texts[0] == f"AUC={roc_auc.mean():.2f}±{roc_auc.std():.2f}"
        assert len(legend_texts) == 2
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

    display = report.metrics.roc()
    display.plot()


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        (
            "comparison_cross_validation_reports_binary_classification",
            ["None", "auto", "estimator"],
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
            [(None, 0), ("estimator", 2)],
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
        facet = display.plot(subplot_by=subplot_by)
        ax_ = facet.axes.flatten()
        if subplot_by is None:
            ax = facet.axes.squeeze().item()
            assert isinstance(ax, mpl.axes.Axes)
        else:
            assert len(ax_) == expected_len
