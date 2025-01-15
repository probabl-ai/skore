"""Configure logging and global settings."""

import logging

from rich import jupyter
from rich.console import Console
from rich.theme import Theme

from skore.project import Project, open
from skore.sklearn import CrossValidationReporter, EstimatorReport, train_test_split
from skore.utils._polyfill import patched_display
from skore.utils._show_versions import show_versions

__all__ = [
    "CrossValidationReporter",
    "EstimatorReport",
    "open",
    "Project",
    "show_versions",
    "train_test_split",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Default to no output
logger.setLevel(logging.INFO)

# Patch the display function for having jupyter notebook style for ipywidgets
# in VS Code
jupyter.display = patched_display

skore_console_theme = Theme(
    {
        "repr.str": "cyan",
        "rule.line": "orange1",
        "repr.url": "orange1",
    }
)

console = Console(theme=skore_console_theme, width=79)
