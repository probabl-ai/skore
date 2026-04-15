import matplotlib as mpl
import numpy as np
import pytest

from skore import ComparisonReport, CrossValidationReport


@pytest.mark.parametrize("subplot_by", [None, "estimator", "auto", "invalid"])
@pytest.mark.parametrize(
    "fixture_name",
    [
        "comparison_cross_validation_reports_binary_classification",
        "comparison_cross_validation_reports_multiclass_classification",
    ],
)
def test_subplot_by(pyplot, subplot_by, fixture_name, request):
    """Check that the subplot_by parameter works correctly for comparison reports."""
    report = request.getfixturevalue(fixture_name)
    display = report.metrics.confusion_matrix()
    if subplot_by in ["invalid", None]:
        err_msg = (
            "Invalid `subplot_by` parameter. Valid options are: estimator or auto. "
            f"Got '{subplot_by}' instead."
        )
        with pytest.raises(ValueError, match=err_msg):
            display.plot(subplot_by=subplot_by)
    elif subplot_by in ["estimator", "auto"]:
        fig = display.plot(subplot_by=subplot_by)
        axes = fig.axes
        assert isinstance(axes[0], mpl.axes.Axes)
        assert len(axes) == len(report.reports_)


def test_split_aggregation(
    pyplot,
    comparison_cross_validation_reports_binary_classification,
    comparison_cross_validation_reports_binary_classification_figure_axes,
):
    """Check that confusion matrix values are aggregated across splits per estimator."""
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.confusion_matrix()
    _, axes = comparison_cross_validation_reports_binary_classification_figure_axes

    for ax in axes:
        annotation_texts = [
            text.get_text()
            for text in ax.texts
            if text.get_text() and "±" in text.get_text()
        ]
        assert len(annotation_texts) == len(display.labels) ** 2

        for text_content in annotation_texts:
            assert "\n" in text_content
            assert "±" in text_content
            parts = text_content.split("\n")
            assert len(parts) == 2
            assert parts[1].startswith("(±")
            assert parts[1].endswith(")")


def test_estimator_names_in_confusion_matrix(
    comparison_cross_validation_reports_binary_classification,
):
    """Check that estimator names are correctly recorded in the confusion matrix."""
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.confusion_matrix()

    estimator_names = display.confusion_matrix_predict["estimator"].unique()
    assert len(estimator_names) == 2
    assert set(estimator_names) == set(report.reports_.keys())


def test_pos_label(pyplot, forest_binary_classification_data):
    """Check that the report_pos_label parameter works correctly."""
    estimator, X, y = forest_binary_classification_data
    labels = np.array(["A", "B"], dtype=object)
    y_labeled = labels[y]

    cv_report_1 = CrossValidationReport(
        estimator,
        X=X,
        y=y_labeled,
        splitter=3,
        pos_label="A",
    )
    cv_report_2 = CrossValidationReport(
        estimator,
        X=X,
        y=y_labeled,
        splitter=3,
        pos_label="A",
    )
    report = ComparisonReport([cv_report_1, cv_report_2])

    display = report.metrics.confusion_matrix()
    fig = display.plot()
    axes = fig.axes
    for idx in range(len(report.reports_)):
        assert axes[idx].get_xticklabels()[1].get_text() == "A*"
    # Only the first subplot has yticklabels
    assert axes[0].get_yticklabels()[1].get_text() in "A*"

    cv_report_1 = CrossValidationReport(
        estimator,
        X=X,
        y=y_labeled,
        splitter=3,
        pos_label="B",
    )
    cv_report_2 = CrossValidationReport(
        estimator,
        X=X,
        y=y_labeled,
        splitter=3,
        pos_label="B",
    )
    report = ComparisonReport([cv_report_1, cv_report_2])
    display = report.metrics.confusion_matrix()
    fig = display.plot()
    axes = fig.axes
    for idx in range(len(report.reports_)):
        assert axes[idx].get_xticklabels()[1].get_text() == "B*"
    assert axes[0].get_yticklabels()[1].get_text() == "B*"
