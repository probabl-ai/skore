import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports_",
        "cross_validation_reports_",
        "comparison_estimator_reports_",
        "comparison_cross_validation_reports_",
    ],
)
@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_frame_structure(pyplot, fixture_prefix, task, request):
    """Check that the frame method returns a properly structured dataframe."""
    report = request.getfixturevalue(fixture_prefix + task + "_classification")
    if isinstance(report, tuple):
        report = report[0]

    display = report.metrics.confusion_matrix()
    n_classes = len(display.display_labels)
    n_splits = 5 if "cross_validation" in fixture_prefix else 1
    n_reports = 2 if "comparison" in fixture_prefix else 1

    frame = display.frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (n_classes * n_classes * n_splits * n_reports, 7)

    expected_columns = [
        "true_label",
        "predicted_label",
        "value",
        "threshold",
        "split",
        "estimator",
        "data_source",
    ]
    assert frame.columns.tolist() == expected_columns
    assert set(frame["true_label"].unique()) == set(display.display_labels)
    assert set(frame["predicted_label"].unique()) == set(display.display_labels)
    assert frame["split"].nunique() == (
        5 if "cross_validation" in fixture_prefix else 0
    )


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports_",
        "cross_validation_reports_",
        "comparison_estimator_reports_",
        "comparison_cross_validation_reports_",
    ],
)
@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_facet_grid_kwargs(pyplot, fixture_prefix, task, request):
    """Check that we can override default facet grid kwargs."""
    report = request.getfixturevalue(fixture_prefix + task + "_classification")
    if isinstance(report, tuple):
        report = report[0]
    display = report.metrics.confusion_matrix()
    display.plot()
    assert display.figure_.get_figheight() == 6

    display.plot(facet_grid_kwargs={"height": 8})
    assert display.figure_.get_figheight() == 8


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports_",
        "cross_validation_reports_",
        "comparison_estimator_reports_",
        "comparison_cross_validation_reports_",
    ],
)
@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_heatmap_kwargs(pyplot, fixture_prefix, task, request):
    """Check that default heatmap kwargs are applied correctly."""
    report = request.getfixturevalue(fixture_prefix + task + "_classification")
    if isinstance(report, tuple):
        report = report[0]

    def get_ax(display):
        return display.ax_[0] if isinstance(display.ax_, np.ndarray) else display.ax_

    n_base_elements = 1 if task == "binary" else 0
    n_plots = 2 if "comparison" in fixture_prefix else 1

    display = report.metrics.confusion_matrix()
    display.plot()
    assert get_ax(display).collections[0].get_cmap().name == "Blues"
    display.plot(heatmap_kwargs={"cmap": "Reds"})
    assert get_ax(display).collections[0].get_cmap().name == "Reds"

    display = report.metrics.confusion_matrix()
    display.plot()
    assert len(get_ax(display).texts) > 1
    display.plot(heatmap_kwargs={"annot": False})
    # There is still the pos_label annotation
    assert len(get_ax(display).texts) == n_base_elements

    display = report.metrics.confusion_matrix()
    display.plot(normalize="all")
    for text in get_ax(display).texts:
        text_content = text.get_text()
        assert "." in text_content or "*" in text_content
    display.plot(normalize="all", heatmap_kwargs={"fmt": ".2e"})
    for text in get_ax(display).texts:
        text_content = text.get_text()
        assert "e" in text_content

    display = report.metrics.confusion_matrix()
    display.plot()
    assert len(display.figure_.axes) == n_plots
    display.plot(heatmap_kwargs={"cbar": True})
    assert len(display.figure_.axes) == 2 * n_plots


@pytest.mark.parametrize(
    "fixture_name",
    [
        "estimator_reports_binary_classification",
        "cross_validation_reports_binary_classification",
        "comparison_estimator_reports_binary_classification",
        "comparison_cross_validation_reports_binary_classification",
    ],
)
def test_thresholds_available_for_binary_classification(pyplot, fixture_name, request):
    """Check that thresholds are available for binary classification."""
    report = request.getfixturevalue(fixture_name)
    if isinstance(report, tuple):
        report = report[0]
    display = report.metrics.confusion_matrix()

    assert display.thresholds is not None
    assert len(display.thresholds) > 0
    assert "threshold" in display.confusion_matrix.columns


@pytest.mark.parametrize(
    "fixture_name",
    [
        "estimator_reports_multiclass_classification",
        "cross_validation_reports_multiclass_classification",
        "comparison_estimator_reports_multiclass_classification",
        "comparison_cross_validation_reports_multiclass_classification",
    ],
)
def test_thresholds_in_multiclass(pyplot, fixture_name, request):
    """Check that the absence of thresholds in handled properly in multiclass."""
    report = request.getfixturevalue(fixture_name)
    if isinstance(report, tuple):
        report = report[0]
    display = report.metrics.confusion_matrix()

    assert len(display.thresholds) == 1
    assert np.isnan(display.thresholds[0])

    err_msg = "Threshold support is only available for binary classification."
    with pytest.raises(ValueError, match=err_msg):
        display.frame(threshold_value=0.5)
    with pytest.raises(ValueError, match=err_msg):
        display.plot(threshold_value=0.5)
