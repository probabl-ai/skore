import matplotlib as mpl
import pytest


def test_legend_binary_classification(
    pyplot,
    cross_validation_reports_binary_classification,
    cross_validation_reports_binary_classification_figure_axes,
):
    """Check the legend of the precision-recall curve plot with binary data."""
    report = cross_validation_reports_binary_classification[0]
    display = report.metrics.precision_recall()
    _, ax = cross_validation_reports_binary_classification_figure_axes
    assert isinstance(ax, mpl.axes.Axes)
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    plot_data = display.frame(with_average_precision=True)
    average_precision = plot_data["average_precision"]
    assert (
        legend_texts[0] == f"AP={average_precision.mean():.2f}"
        f"±{average_precision.std():.2f}"
    )
    assert len(legend_texts) == 1


def test_legend_multiclass_classification(
    pyplot,
    cross_validation_reports_multiclass_classification,
    cross_validation_reports_multiclass_classification_figure_axes,
):
    """Check the legend of the precision-recall curve plot with multiclass data."""
    report = cross_validation_reports_multiclass_classification[0]
    display = report.metrics.precision_recall()
    _, ax = cross_validation_reports_multiclass_classification_figure_axes
    ax = ax[0]
    labels = display.precision_recall["label"].cat.categories
    assert isinstance(ax, mpl.axes.Axes)
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    for label_idx, label in enumerate(labels):
        plot_data = display.frame(with_average_precision=True)
        average_precision = plot_data.query(f"label == {label}")["average_precision"]
        assert (
            legend_texts[label_idx] == f"{label} (AP={average_precision.mean():.2f}"
            f"±{average_precision.std():.2f})"
        )
    assert len(legend_texts) == len(labels)


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        ("cross_validation_reports_binary_classification", ["None", "auto"]),
        (
            "cross_validation_reports_multiclass_classification",
            ["None", "auto", "label"],
        ),
    ],
)
def test_invalid_subplot_by(fixture_name, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    report = request.getfixturevalue(fixture_name)[0]

    display = report.metrics.precision_recall()
    err_msg = (
        f"subplot_by must be one of {', '.join(valid_values)}. Got 'invalid' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "fixture_name, subplot_by_tuples",
    [
        ("cross_validation_reports_binary_classification", [(None, 0)]),
        (
            "cross_validation_reports_multiclass_classification",
            [("label", 3)],
        ),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by_tuples, request):
    """Check that we can pass non default values to `subplot_by`."""
    report = request.getfixturevalue(fixture_name)[0]
    display = report.metrics.precision_recall()
    for subplot_by, expected_len in subplot_by_tuples:
        facet = display.plot(subplot_by=subplot_by)
        if subplot_by is None:
            ax = facet.axes.squeeze().item()
            assert isinstance(ax, mpl.axes.Axes)
        else:
            ax_ = facet.axes.flatten()
            assert len(ax_) == expected_len
