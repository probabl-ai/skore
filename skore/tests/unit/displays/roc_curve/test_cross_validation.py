import matplotlib as mpl
import pytest


def test_legend_binary_classification(
    pyplot,
    cross_validation_reports_binary_classification,
    cross_validation_reports_binary_classification_figure_axes,
):
    """Check the legend of the ROC curve plot with binary data."""
    report = cross_validation_reports_binary_classification[0]
    display = report.metrics.roc()
    _, ax = cross_validation_reports_binary_classification_figure_axes
    legend = ax[0].get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    labels = display.roc_curve["label"].cat.categories
    for label_idx, label in enumerate(labels):
        plot_data = display.frame(with_roc_auc=True)
        roc_auc = plot_data.query(f"label == {label}")["roc_auc"]
        assert (
            legend_texts[label_idx] == f"{label} (AUC={roc_auc.mean():.2f}"
            f"±{roc_auc.std():.2f})"
        )
    assert len(legend_texts) == len(labels) + 1
    assert legend_texts[-1] == "Chance level (AUC = 0.5)"


def test_legend_multiclass_classification(
    pyplot,
    cross_validation_reports_multiclass_classification,
    cross_validation_reports_multiclass_classification_figure_axes,
):
    """Check the legend of the ROC curve plot with multiclass data."""
    report = cross_validation_reports_multiclass_classification[0]
    display = report.metrics.roc()
    _, ax = cross_validation_reports_multiclass_classification_figure_axes
    labels = display.roc_curve["label"].cat.categories
    legend = ax[0].get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    for label_idx, label in enumerate(labels):
        plot_data = display.frame(with_roc_auc=True)
        roc_auc = plot_data.query(f"label == {label}")["roc_auc"]
        assert (
            legend_texts[label_idx] == f"{label} (AUC={roc_auc.mean():.2f}"
            f"±{roc_auc.std():.2f})"
        )
    assert len(legend_texts) == len(labels) + 1
    assert legend_texts[-1] == "Chance level (AUC = 0.5)"


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        ("cross_validation_reports_binary_classification", ["None", "auto", "label"]),
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

    display = report.metrics.roc()
    err_msg = (
        f"subplot_by must be one of {', '.join(valid_values)}. Got 'invalid' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "fixture_name, subplot_by_tuples",
    [
        ("cross_validation_reports_binary_classification", [(None, 0), ("label", 2)]),
        (
            "cross_validation_reports_multiclass_classification",
            [("label", 3)],
        ),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by_tuples, request):
    """Check that we can pass non default values to `subplot_by`."""
    report = request.getfixturevalue(fixture_name)[0]
    display = report.metrics.roc()
    for subplot_by, expected_len in subplot_by_tuples:
        fig = display.plot(subplot_by=subplot_by)
        axes = fig.axes
        if subplot_by is None:
            assert len(axes) == 1
            assert isinstance(axes[0], mpl.axes.Axes)
        else:
            assert len(axes) == expected_len
