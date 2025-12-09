"""Configure logging and global settings."""

import logging
import warnings

import joblib
from rich.console import Console
from rich.theme import Theme

from skore._config import config_context, get_config, set_config
from skore._externals._sklearn_compat import parse_version
from skore._sklearn import (
    ComparisonReport,
    ConfusionMatrixDisplay,
    CrossValidationReport,
    EstimatorReport,
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
    TableReportDisplay,
    train_test_split,
)
from skore._sklearn._plot.base import Display
from skore._sklearn._plot.metrics.feature_importance_coefficients_display import (
    FeatureImportanceCoefficientsDisplay,
)
from skore._utils._patch import setup_jupyter_display
from skore._utils._show_versions import show_versions
from skore.project import Project

__all__ = [
    "CrossValidationReport",
    "ComparisonReport",
    "ConfusionMatrixDisplay",
    "Display",
    "EstimatorReport",
    "PrecisionRecallCurveDisplay",
    "PredictionErrorDisplay",
    "Project",
    "RocCurveDisplay",
    "MetricsSummaryDisplay",
    "show_versions",
    "train_test_split",
    "config_context",
    "get_config",
    "set_config",
    "TableReportDisplay",
    "FeatureImportanceCoefficientsDisplay",
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

joblib_version = parse_version(joblib.__version__)
if joblib_version < parse_version("1.4"):
    warnings.warn(
        "Because your version of joblib is older than 1.4, some of the features of "
        "skore will not be enabled (e.g. progress bars). You can update joblib to "
        "benefit from these features.",
        stacklevel=2,
    )
    set_config(show_progress=False)
