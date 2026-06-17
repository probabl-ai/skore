"""Tests for DisplayMixin default repr."""


def test_default_repr_includes_frame_and_hint(repr_coefficients_display):
    display = repr_coefficients_display
    repr_str = repr(display)
    assert repr(display.frame()) in repr_str
    assert repr_str.endswith(
        "Use .plot() to plot the data and .frame() to access the full data."
    )


def test_default_repr_html_includes_plot_and_hint(pyplot, repr_coefficients_display):
    display = repr_coefficients_display
    html = display._repr_html_()
    assert "data:image/png;base64," in html
    assert "Use <code>.plot()</code> to control the view and" in html
    assert "<code>.frame()</code> to access the plotted data." in html


def test_default_repr_mimebundle(pyplot, repr_coefficients_display):
    bundle = repr_coefficients_display._repr_mimebundle_()
    assert bundle == {
        "text/plain": repr(repr_coefficients_display),
        "text/html": repr_coefficients_display._repr_html_(),
    }
