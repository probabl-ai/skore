"""Tests for MetricsSummaryDisplay repr."""

from skore import EstimatorReport
from skore._sklearn._plot.metrics.metrics_summary_display import frame_repr_html


def _frame_html(frame):
    return frame_repr_html(frame)


def test_repr_includes_frame_and_hint(forest_binary_classification_with_test):
    """Check that __repr__ shows the default frame and a trailing hint."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    display = EstimatorReport(
        estimator, X_test=X_test, y_test=y_test
    ).metrics.summarize()

    repr_str = repr(display)
    assert repr_str.startswith(repr(display._repr_frame()))
    assert repr_str.endswith("Use .frame() to control the format of the output.")
    assert (
        "Use .plot() to plot the data" not in display._repr_mimebundle_()["text/plain"]
    )


def test_repr_html_includes_frame_and_hint(forest_binary_classification_with_test):
    """Check that _repr_html_ shows the default frame and a trailing hint."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    display = EstimatorReport(
        estimator, X_test=X_test, y_test=y_test
    ).metrics.summarize()

    html = display._repr_html_()
    assert html.startswith(_frame_html(display._repr_frame(for_html=True)))
    assert "Use <code>.frame()</code> to control the format of the output." in html
    mime_html = display._repr_mimebundle_()["text/html"]
    assert "data:image/png;base64," not in mime_html
    assert "Use <code>.plot()</code> to control the view" not in mime_html


def test_repr_frame_for_html_uses_verbose_names_and_multiindex(
    forest_multiclass_classification_with_test,
):
    """HTML repr uses auto format with verbose names and a MultiIndex."""
    import pandas as pd

    estimator, X_test, y_test = forest_multiclass_classification_with_test
    display = EstimatorReport(
        estimator, X_test=X_test, y_test=y_test
    ).metrics.summarize()

    html_frame = display._repr_frame(for_html=True)
    plain_frame = display._repr_frame(for_html=False)

    assert isinstance(html_frame.index, pd.MultiIndex)
    assert "Metric" in html_frame.index.names
    assert "Accuracy" in html_frame.index.get_level_values("Metric")
    assert "accuracy" not in html_frame.index.get_level_values("Metric")
    assert "accuracy" in plain_frame.index
