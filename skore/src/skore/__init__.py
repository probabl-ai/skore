"""Configure logging and global settings."""

import logging

from rich.console import Console
from rich.theme import Theme

from skore.project import Project
from skore.sklearn import (
    CrossValidationReport,
    CrossValidationReporter,
    EstimatorReport,
    train_test_split,
)
from skore.utils._patch import setup_jupyter_display
from skore.utils._show_versions import show_versions

__all__ = [
    "CrossValidationReporter",
    "CrossValidationReport",
    "EstimatorReport",
    "Project",
    "show_versions",
    "train_test_split",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Default to no output
logger.setLevel(logging.INFO)

# Configure jupyter display for VS Code compatibility
setup_jupyter_display()

skore_console_theme = Theme(
    {
        "repr.str": "cyan",
        "rule.line": "orange1",
        "repr.url": "orange1",
    }
)

console = Console(theme=skore_console_theme, width=88)
