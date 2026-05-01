"""Configure logging and global settings."""

from importlib.metadata import version
from logging import INFO, NullHandler, getLogger
from warnings import warn

from joblib import __version__ as joblib_version
from matplotlib import pyplot as plt
from rich.console import Console
from rich.theme import Theme

from skore._config import configuration
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
    TrainTestSplit,
    compare,
    evaluate,
    train_test_split,
)
from skore._sklearn._diagnostic import Check, CheckNotApplicable, DiagnosticDisplay
from skore._sklearn._plot.base import Display
from skore._sklearn._plot.inspection.calibration_curve import (
    CalibrationDisplay,
)
from skore._sklearn._plot.inspection.coefficients import CoefficientsDisplay
from skore._sklearn._plot.inspection.impurity_decrease import (
    ImpurityDecreaseDisplay,
)
from skore._sklearn._plot.inspection.permutation_importance import (
    PermutationImportanceDisplay,
)
from skore._utils._environment import is_environment_notebook_like
from skore._utils._patch import setup_jupyter_display
from skore._utils._show_versions import show_versions

plt.ion()
setup_jupyter_display()


if parse_version(joblib_version) < parse_version("1.4"):
    configuration.show_progress = False
    warn(
        "Because your version of joblib is older than 1.4, some of the features of "
        "skore will not be enabled (e.g. progress bars). You can update joblib to "
        "benefit from these features.",
        stacklevel=2,
    )


__version__ = version("skore")
__all__ = [
    "Check",
    "CheckNotApplicable",
    "CoefficientsDisplay",
    "ComparisonReport",
    "ConfusionMatrixDisplay",
    "CrossValidationReport",
    "DiagnosticDisplay",
    "Display",
    "EstimatorReport",
    "ImpurityDecreaseDisplay",
    "MetricsSummaryDisplay",
    "PermutationImportanceDisplay",
    "PrecisionRecallCurveDisplay",
    "PredictionErrorDisplay",
    "Project",
    "RocCurveDisplay",
    "CalibrationDisplay",
    "THREADABLE",
    "TableReportDisplay",
    "TrainTestSplit",
    "compare",
    "configuration",
    "console",
    "evaluate",
    "login",
    "show_versions",
    "train_test_split",
]


logger = getLogger(__name__)
logger.addHandler(NullHandler())  # Default to no output
logger.setLevel(INFO)


console = Console(
    width=88,
    theme=Theme({"repr.str": "cyan", "rule.line": "orange1", "repr.url": "orange1"}),
    # FIXME:
    # Force `force_jupyter` on Jupyterlite.
    # Waiting for the merge of https://github.com/Textualize/rich/pull/4104.
    force_jupyter=(is_environment_notebook_like() or None),
)


try:
    from threading import Thread

    thread = Thread()
    thread.start()
    thread.join()
except Exception:
    THREADABLE = False
else:
    THREADABLE = True
