"""Configure logging and global settings."""

from logging import INFO, NullHandler, getLogger
from warnings import warn

from joblib import __version__ as joblib_version
from rich.console import Console
from rich.theme import Theme

from skore._config import config_context, get_config, set_config
from skore._externals._sklearn_compat import parse_version
from skore._project.login import login
from skore._project.project import Project
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
from skore._sklearn._plot.inspection.coefficients import CoefficientsDisplay
from skore._sklearn._plot.inspection.impurity_decrease import (
    ImpurityDecreaseDisplay,
)
from skore._sklearn._plot.inspection.permutation_importance import (
    PermutationImportanceDisplay,
)
from skore._utils._patch import setup_jupyter_display
from skore._utils._show_versions import show_versions

# Configure jupyter display for VS Code compatibility
setup_jupyter_display()


if parse_version(joblib_version) < parse_version("1.4"):
    set_config(show_progress=False)
    warn(
        "Because your version of joblib is older than 1.4, some of the features of "
        "skore will not be enabled (e.g. progress bars). You can update joblib to "
        "benefit from these features.",
        stacklevel=2,
    )


__all__ = [
    "CoefficientsDisplay",
    "ComparisonReport",
    "ConfusionMatrixDisplay",
    "CrossValidationReport",
    "Display",
    "EstimatorReport",
    "ImpurityDecreaseDisplay",
    "MetricsSummaryDisplay",
    "PermutationImportanceDisplay",
    "PrecisionRecallCurveDisplay",
    "PredictionErrorDisplay",
    "Project",
    "RocCurveDisplay",
    "TableReportDisplay",
    "config_context",
    "get_config",
    "login",
    "set_config",
    "show_versions",
    "train_test_split",
]


logger = getLogger(__name__)
logger.addHandler(NullHandler())  # Default to no output
logger.setLevel(INFO)


skore_console_theme = Theme(
    {
        "repr.str": "cyan",
        "rule.line": "orange1",
        "repr.url": "orange1",
    }
)


console = Console(theme=skore_console_theme, width=88)
