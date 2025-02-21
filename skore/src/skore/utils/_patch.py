import os
from collections.abc import Iterable

from rich import jupyter
from rich.segment import Segment

_original_render_segments = jupyter._render_segments


def patched_render_segments(segments: Iterable[Segment]) -> str:
    """Patched version of rich.jupyter._render_segments that includes VS Code styling.

    This is to make sure that the CSS style exposed by jupyter notebook and used
    by ipywidgets is applied to rich output.
    Currently, the VS Code extension does not support the CSS style exposed by jupyter
    notebook and used by ipywidgets.
    We should track the progress in the following issue:
    https://github.com/microsoft/vscode-jupyter/issues/7161

    """
    html = _original_render_segments(segments)
    # Apply VS Code styling if we're in VS Code
    if "VSCODE_PID" in os.environ:
        css = """
        <style>
        .cell-output-ipywidget-background {
            background-color: transparent !important;
        }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground);
            --jp-widgets-font-size: var(--vscode-editor-font-size);
        }
        </style>
        """
        html += css

    return html


def setup_jupyter_display() -> None:
    """Configure the jupyter display to work properly in VS Code."""
    jupyter._render_segments = patched_render_segments
