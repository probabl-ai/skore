"""Tests for MetricsSummaryDisplay repr."""

import pandas as pd

from skore import EstimatorReport


def test_repr_includes_frame_and_hint(forest_binary_classification_with_test):
    """Check that __repr__ shows the default frame and a trailing hint."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    display = EstimatorReport(
        estimator, X_test=X_test, y_test=y_test
    ).metrics.summarize()

    repr_str = repr(display)
    assert repr_str.startswith(repr(display.frame(format="auto")))
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
    frame = display.frame(format="auto", verbose_name=True, flat_index=False)
    expected_html = (
        frame.to_frame()._repr_html_()
        if isinstance(frame, pd.Series)
        else frame._repr_html_()
    )
    assert html.startswith(expected_html)
    assert "Use <code>.frame()</code> to control the format of the output." in html
    mime_html = display._repr_mimebundle_()["text/html"]
    assert "data:image/png;base64," not in mime_html
    assert "Use <code>.plot()</code> to control the view" not in mime_html


def test_repr_frame_for_html_uses_verbose_names_and_multiindex(
    forest_multiclass_classification_with_test,
):
    """HTML repr uses auto format with verbose names and a MultiIndex."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    display = EstimatorReport(
        estimator, X_test=X_test, y_test=y_test
    ).metrics.summarize()

    html_frame = display.frame(format="auto", verbose_name=True, flat_index=False)
    plain_frame = display.frame(format="auto", flat_index=True)

    assert isinstance(html_frame.index, pd.MultiIndex)
    assert "Metric" in html_frame.index.names
    assert "Output" not in html_frame.index.names
    assert "Accuracy" in html_frame.index.get_level_values("Metric")
    assert "accuracy" not in html_frame.index.get_level_values("Metric")
    assert "accuracy" in plain_frame.index
