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
    Check the legend of the precision-recall curve plot with binary data.
    """
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.precision_recall()
    _, axes = comparison_cross_validation_reports_binary_classification_figure_axes
    for ax, estimator in zip(axes, report.reports_, strict=True):
        assert isinstance(ax, mpl.axes.Axes)
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        plot_data = display.frame(with_average_precision=True)
        average_precision = plot_data.query(f"estimator == '{estimator}'")[
            "average_precision"
        ]
        assert (
            legend_texts[0] == f"AP={average_precision.mean():.2f}"
            f"±{average_precision.std():.2f}"
        )
        assert len(legend_texts) == 1


def test_legend_multiclass_classification(
    pyplot,
    comparison_cross_validation_reports_multiclass_classification,
    comparison_cross_validation_reports_multiclass_classification_figure_axes,
):
    """
    Check the legend of the precision-recall curve plot with multiclass data.
    """
    report = comparison_cross_validation_reports_multiclass_classification
    display = report.metrics.precision_recall()
    _, axes = comparison_cross_validation_reports_multiclass_classification_figure_axes

    labels = display.precision_recall["label"].cat.categories

    for ax, estimator in zip(axes, report.reports_, strict=True):
        assert isinstance(ax, mpl.axes.Axes)
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        for label_idx, label in enumerate(labels):
            plot_data = display.frame(with_average_precision=True)
            average_precision = plot_data.query(
                f"label == {label} & estimator == '{estimator}'"
            )["average_precision"]
            assert (
                legend_texts[label_idx] == f"{label} (AP={average_precision.mean():.2f}"
                f"±{average_precision.std():.2f})"
            )


def test_multiclass_str_labels_precision_recall_plot(pyplot):
    """Regression test for issue #2183 with multiclass comparison reports.

    Using string labels backed by numpy.str_ should not break
    `precision_recall().plot()` for a multiclass ComparisonReport.
    """
    iris = load_iris(as_frame=True)
    X, y_int = iris.data, iris.target
    y = iris.target_names[y_int]

    report_1 = CrossValidationReport(LogisticRegression(max_iter=500), X=X, y=y)
    report_2 = CrossValidationReport(LogisticRegression(max_iter=1000), X=X, y=y)
    report = ComparisonReport([report_1, report_2])

    display = report.metrics.precision_recall()
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
    display = report.metrics.precision_recall()
    valid_values_str = ", ".join(valid_values)
    err_msg = f"subplot_by must be one of {valid_values_str}. Got 'invalid' instead."
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
    display = report.metrics.precision_recall()
    for subplot_by, expected_len in subplot_by_tuples:
        display.plot(subplot_by=subplot_by)
        if subplot_by is None:
            assert isinstance(display.ax_, mpl.axes.Axes)
        else:
            assert len(display.ax_) == expected_len
