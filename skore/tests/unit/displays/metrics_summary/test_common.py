"""Tests for MetricsSummaryDisplay repr."""

from skore import EstimatorReport

_FRAME_HINT = "Use .frame() to control the format of the output."
_FRAME_HINT_HTML = "<code>.frame()</code>"


def test_repr_includes_frame_and_hint(forest_binary_classification_with_test):
    """Check that __repr__ shows the default frame and a trailing hint."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    display = EstimatorReport(
        estimator, X_test=X_test, y_test=y_test
    ).metrics.summarize()

    repr_str = repr(display)
    assert repr_str.startswith(repr(display.frame()))
    assert repr_str.endswith(_FRAME_HINT)


def test_repr_html_includes_frame_and_hint(forest_binary_classification_with_test):
    """Check that _repr_html_ shows the default frame and a trailing hint."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    display = EstimatorReport(
        estimator, X_test=X_test, y_test=y_test
    ).metrics.summarize()

    html = display._repr_html_()
    assert html.startswith(display.frame()._repr_html_())
    assert _FRAME_HINT_HTML in html


def test_repr_mimebundle(forest_binary_classification_with_test):
    """Check that _repr_mimebundle_ returns text/plain and text/html."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    display = EstimatorReport(
        estimator, X_test=X_test, y_test=y_test
    ).metrics.summarize()

    bundle = display._repr_mimebundle_()
    assert bundle["text/plain"] == repr(display)
    assert bundle["text/html"] == display._repr_html_()
