"""Configure logging and global settings."""

import logging
import os

from rich import jupyter
from rich.console import Console
from rich.theme import Theme

from skore.project import Project, open
from skore.sklearn import (
    CrossValidationReport,
    CrossValidationReporter,
    EstimatorReport,
    train_test_split,
)
from skore.utils._show_versions import show_versions

__all__ = [
    "CrossValidationReporter",
    "CrossValidationReport",
    "EstimatorReport",
    "open",
    "Project",
    "show_versions",
    "train_test_split",
]

########################################################################################
# FIXME: This is a temporary patch to make the output of the Jupyter notebook look nice.
# We should find a better solution in the future.
########################################################################################

# Store the original display function
original_display = jupyter.display


def patched_display(segments, text):
    """Patched version of rich.jupyter.display that includes VS Code styling."""
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


# Patch the display function
jupyter.display = patched_display

########################################################################################
# End of the temporary patch
########################################################################################


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Default to no output
logger.setLevel(logging.INFO)

skore_console_theme = Theme(
    {
        "repr.str": "cyan",
        "rule.line": "orange1",
        "repr.url": "orange1",
    }
)

console = Console(theme=skore_console_theme, width=79)
