"""Configure logging and global settings."""

import logging

from rich.console import Console
from rich.theme import Theme

from skore._config import config_context, get_config, set_config
from skore.project import Project
from skore.sklearn import (
    ComparisonReport,
    CrossValidationReport,
    EstimatorReport,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
    train_test_split,
)
from skore.utils._patch import setup_jupyter_display
from skore.utils._show_versions import show_versions

__all__ = [
    "CrossValidationReport",
    "ComparisonReport",
    "EstimatorReport",
    "PrecisionRecallCurveDisplay",
    "PredictionErrorDisplay",
    "Project",
    "RocCurveDisplay",
    "show_versions",
    "train_test_split",
    "config_context",
    "get_config",
    "set_config",
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
