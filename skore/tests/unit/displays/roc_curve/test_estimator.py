import matplotlib as mpl
import pytest


def test_legend_binary_classification(
    pyplot,
    estimator_reports_binary_classification,
    estimator_reports_binary_classification_figure_axes,
):
    """Check the legend of the ROC curve plot with binary data."""
    report = estimator_reports_binary_classification[0]
    display = report.metrics.roc()
    _, ax = estimator_reports_binary_classification_figure_axes
    assert isinstance(ax, mpl.axes.Axes)
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    plot_data = display.frame(with_roc_auc=True)
    roc_auc = plot_data["roc_auc"].iloc[0]
    assert legend_texts[0] == f"AUC={roc_auc:.2f}"
    assert len(legend_texts) == 2
    assert legend_texts[-1] == "Chance level (AUC = 0.5)"


def test_legend_multiclass_classification(
    pyplot,
    estimator_reports_multiclass_classification,
    estimator_reports_multiclass_classification_figure_axes,
):
    """Check the legend of the ROC curve plot with multiclass data."""
    report = estimator_reports_multiclass_classification[0]
    display = report.metrics.roc()
    _, ax = estimator_reports_multiclass_classification_figure_axes
    labels = display.roc_curve["label"].cat.categories

    assert isinstance(ax, mpl.axes.Axes)
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    for label_idx, label in enumerate(labels):
        plot_data = display.frame(with_roc_auc=True)
        roc_auc = plot_data.query(f"label == {label}")["roc_auc"].iloc[0]
        assert legend_texts[label_idx] == f"{label} (AUC={roc_auc:.2f})"
    assert len(legend_texts) == len(labels) + 1
    assert legend_texts[-1] == "Chance level (AUC = 0.5)"


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        ("estimator_reports_binary_classification", ["None", "auto"]),
        (
            "estimator_reports_multiclass_classification",
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
        ("estimator_reports_binary_classification", [(None, 0)]),
        ("estimator_reports_multiclass_classification", [("label", 3)]),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by_tuples, request):
    """Check that we can pass non default values to `subplot_by`."""
    report = request.getfixturevalue(fixture_name)[0]
    display = report.metrics.roc()
    for subplot_by, expected_len in subplot_by_tuples:
        display.plot(subplot_by=subplot_by)
        if subplot_by is None:
            assert isinstance(display.ax_, mpl.axes.Axes)
        else:
            assert len(display.ax_) == expected_len


@pytest.mark.parametrize(
    "fixture_name",
    [
        "estimator_reports_binary_classification",
        "estimator_reports_multiclass_classification",
    ],
)
def test_source_both(pyplot, fixture_name, request):
    """Check the behaviour of the plot when data_source='both'."""
    report = request.getfixturevalue(fixture_name)[0]
    display = report.metrics.roc(data_source="both")
    display.plot()
    ax = display.ax_
    assert isinstance(ax, mpl.axes.Axes)
    assert len(ax.get_lines()) == 3 if "binary" in fixture_name else 7
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    plot_data = display.frame(with_roc_auc=True)
    labels = (
        display.roc_curve["label"].cat.categories
        if display.ml_task == "multiclass-classification"
        else [None]
    )
    expected = []
    for label in labels:
        for data_src in ("train", "test"):
            row = (
                plot_data.query(f"data_source == '{data_src}'")
                if label is None
                else plot_data.query(f"label == {label} & data_source == '{data_src}'")
            )
            auc_val = row["roc_auc"].iloc[0]
            if label is None:
                expected.append(f"{data_src.title()} set (AUC={auc_val:.2f})")
            else:
                expected.append(f"{label} - {data_src.title()} set (AUC={auc_val:.2f})")
    expected.append("Chance level (AUC = 0.5)")
    assert legend_texts == expected
