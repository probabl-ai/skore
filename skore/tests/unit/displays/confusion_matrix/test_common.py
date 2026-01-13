import numpy as np
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
