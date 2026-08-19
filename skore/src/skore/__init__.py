"""Evaluate and compare scikit-learn compatible models with rich reports.

This package provides tools to evaluate estimators, compare models, persist
experiment results, and inspect model behavior through interactive reports.
"""

from importlib.metadata import version
from logging import INFO, NullHandler, getLogger
from warnings import warn

from joblib import __version__ as joblib_version
from matplotlib import pyplot as plt
from rich.console import Console
from rich.theme import Theme

from skore._config import configuration
from skore._externals._sklearn_compat import parse_version
from skore.checks import Check, CheckNotApplicable, ChecksSummaryDisplay
from skore.dispatch import compare, evaluate
from skore.displays import (
    ConfusionMatrixDisplay,
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
    TableReportDisplay,
)
from skore.displays.base import Display
from skore.displays.inspection.calibration_curve import (
    CalibrationDisplay,
)
from skore.displays.inspection.coefficients import CoefficientsDisplay
from skore.displays.inspection.impurity_decrease import (
    ImpurityDecreaseDisplay,
)
from skore.displays.inspection.permutation_importance import (
    PermutationImportanceDisplay,
)
from skore.project._summary import Summary
from skore.project.login import login
from skore.project.project import Project
from skore.reports import ComparisonReport, CrossValidationReport, EstimatorReport
from skore.sklearn import TrainTestSplit
from skore.utils._environment import is_environment_notebook_like
from skore.utils._patch import setup_jupyter_display
from skore.utils._show_versions import show_versions

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
    "ChecksSummaryDisplay",
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
    "Summary",
    "TableReportDisplay",
    "TrainTestSplit",
    "compare",
    "configuration",
    "evaluate",
    "login",
    "show_versions",
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


# Whether threading is available or not.
THREADABLE: bool = True
try:
    from threading import Thread

    thread = Thread()
    thread.start()
    thread.join()
except Exception:
    THREADABLE = False
