import os

from rich import jupyter
from rich.segment import Segment

from skore._utils.patch import patched_render_segments, setup_jupyter_display


def test_patched_render_segments_drops_empty_pre(monkeypatch):
    """Empty progress-bar markup does not leave vertical space behind."""
    monkeypatch.setattr(
        "skore._utils.patch._original_render_segments",
        lambda segments: "<pre> \n </pre>",
    )

    assert patched_render_segments([Segment("unused")]) == ""


def test_patched_render_segments_keeps_content_and_adds_vscode_style(monkeypatch):
    """Non-empty HTML is preserved, with VS Code widget styles when detected."""
    monkeypatch.setattr(
        "skore._utils.patch._original_render_segments",
        lambda segments: "<div>hello</div>",
    )
    monkeypatch.setitem(os.environ, "VSCODE_PID", "123")

    html = patched_render_segments([Segment("hello")])

    assert html.startswith("<div>hello</div>")
    assert "--jp-widgets-color" in html


def test_setup_jupyter_display_installs_patch():
    """The Jupyter renderer is replaced by the skore patch and can be restored."""
    original = jupyter._render_segments
    try:
        setup_jupyter_display()
        assert jupyter._render_segments is patched_render_segments
    finally:
        jupyter._render_segments = original
