import os

from rich import jupyter

original_display = jupyter.display


def patched_display(segments, text):
    """Patched version of rich.jupyter.display that includes VS Code styling.

    This is to make sure that the CSS style exposed by jupyter notebook and used
    by ipywidgets is applied to rich output.
    Currently, the VS Code extension does not support the CSS style exposed by jupyter
    notebook and used by ipywidgets.
    We should track the progress in the following issue:
    https://github.com/microsoft/vscode-jupyter/issues/7161

    """
    # Call the original display function first
    original_display(segments, text)

    # Apply VS Code styling if we're in VS Code
    if "VSCODE_PID" in os.environ:
        from IPython.display import HTML, display

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
        display(HTML(css))
