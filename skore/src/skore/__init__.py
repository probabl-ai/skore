"""Evaluate and compare scikit-learn compatible models with rich reports.

This package provides tools to evaluate estimators, compare models, persist
experiment results, and inspect model behavior through interactive reports.
"""

from importlib import import_module
from importlib.metadata import version
from logging import INFO, NullHandler, getLogger
from typing import TYPE_CHECKING, Any

from matplotlib import pyplot as plt
from rich.console import Console
from rich.theme import Theme

from skore._checks import Check, CheckNotApplicable, ChecksSummaryDisplay
from skore._config import configuration
from skore._dispatch import compare, evaluate
from skore._displays.base import Display
from skore._project.login import login
from skore._project.project import Project
from skore._project.summary import Summary
from skore._reports import ComparisonReport, CrossValidationReport, EstimatorReport
from skore._sklearn import TrainTestSplit
from skore._utils.environment import is_environment_notebook_like
from skore._utils.patch import setup_jupyter_display
from skore._utils.show_versions import show_versions

plt.ion()
setup_jupyter_display()


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


__lazy__ = {
    "CalibrationDisplay": "skore._displays.inspection.calibration_curve",
    "CoefficientsDisplay": "skore._displays.inspection.coefficients",
    "ConfusionMatrixDisplay": "skore._displays.metrics.confusion_matrix",
    "ImpurityDecreaseDisplay": "skore._displays.inspection.impurity_decrease",
    "MetricsSummaryDisplay": "skore._displays.metrics.metrics_summary_display",
    "PermutationImportanceDisplay": (
        "skore._displays.inspection.permutation_importance"
    ),
    "PrecisionRecallCurveDisplay": "skore._displays.metrics.precision_recall_curve",
    "PredictionErrorDisplay": "skore._displays.metrics.prediction_error",
    "RocCurveDisplay": "skore._displays.metrics.roc_curve",
    "TableReportDisplay": "skore._displays.data.table_report",
}

if TYPE_CHECKING:
    from skore._displays.data.table_report import TableReportDisplay
    from skore._displays.inspection.calibration_curve import CalibrationDisplay
    from skore._displays.inspection.coefficients import CoefficientsDisplay
    from skore._displays.inspection.impurity_decrease import ImpurityDecreaseDisplay
    from skore._displays.inspection.permutation_importance import (
        PermutationImportanceDisplay,
    )
    from skore._displays.metrics.confusion_matrix import ConfusionMatrixDisplay
    from skore._displays.metrics.metrics_summary_display import MetricsSummaryDisplay
    from skore._displays.metrics.precision_recall_curve import (
        PrecisionRecallCurveDisplay,
    )
    from skore._displays.metrics.prediction_error import PredictionErrorDisplay
    from skore._displays.metrics.roc_curve import RocCurveDisplay


def __getattr__(name: str) -> Any:
    if module_name := __lazy__.get(name):
        value = getattr(import_module(module_name), name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


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
