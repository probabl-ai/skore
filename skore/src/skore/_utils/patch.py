import os
import re
from collections.abc import Iterable

from rich import jupyter
from rich.segment import Segment

_original_render_segments = jupyter._render_segments


def patched_render_segments(segments: Iterable[Segment]) -> str:
    """Patched version of rich.jupyter._render_segments for Jupyter notebooks.

    This patch applies two fixes:

    1. **VS Code styling support**: Ensures that the CSS style exposed by jupyter
       notebook and used by ipywidgets is applied to rich output. Currently, the VS Code
       extension does not support the CSS style exposed by jupyter notebook and used by
       ipywidgets. We should track the progress in the following issue:
       https://github.com/microsoft/vscode-jupyter/issues/7161

    2. **Empty progress bar cleanup**: Prevents empty <pre> tags from being rendered
       when progress bars are destroyed, which would otherwise create unwanted vertical
       spacing in notebook outputs. When a progress bar completes or is removed, Rich
       may render empty segments that result in empty HTML <pre> tags. This patch checks
       if the rendered HTML only contains an empty <pre> tag and returns an empty string
       instead, preventing vertical space accumulation when multiple progress bars are
       used in loops.
    """
    html = _original_render_segments(segments)

    # Fix 1: Prevent empty <pre> tags from creating unwanted vertical space.
    # This happens when progress bars are destroyed and leave behind empty HTML.
    # Check if the HTML is just an empty <pre> tag (possibly with whitespace)
    if html:
        pre_pattern = r"<pre[^>]*>(.*?)</pre>"
        match = re.search(pre_pattern, html, re.DOTALL)
        if match:
            pre_content = match.group(1)
            # If the content inside the pre tag is empty or only whitespace,
            # return empty string to avoid creating vertical space
            if not pre_content or not pre_content.strip():
                html = ""

    # Fix 2: Apply VS Code styling if we're in VS Code
    if html and "VSCODE_PID" in os.environ:
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
