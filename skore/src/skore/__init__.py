"""Evaluate and compare scikit-learn compatible models with rich reports.

This package provides tools to evaluate estimators, compare models, persist
experiment results, and inspect model behavior through interactive reports.
"""

from importlib.metadata import version
from logging import INFO, NullHandler, getLogger
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from rich.console import Console
from rich.theme import Theme

from skore._externals import lazy_loader as lazy
from skore._utils.environment import is_environment_notebook_like
from skore._utils.patch import setup_jupyter_display

if TYPE_CHECKING:
    from skore._checks import Check, CheckNotApplicable, ChecksSummaryDisplay
    from skore._config import configuration
    from skore._dispatch import compare, evaluate
    from skore._displays import (
        ConfusionMatrixDisplay,
        MetricsSummaryDisplay,
        PrecisionRecallCurveDisplay,
        PredictionErrorDisplay,
        RocCurveDisplay,
        TableReportDisplay,
    )
    from skore._displays.base import Display
    from skore._displays.inspection.calibration_curve import CalibrationDisplay
    from skore._displays.inspection.coefficients import CoefficientsDisplay
    from skore._displays.inspection.impurity_decrease import ImpurityDecreaseDisplay
    from skore._displays.inspection.permutation_importance import (
        PermutationImportanceDisplay,
    )
    from skore._project.login import login
    from skore._project.project import Project
    from skore._project.summary import Summary
    from skore._reports import (
        ComparisonReport,
        CrossValidationReport,
        EstimatorReport,
    )
    from skore._sklearn import TrainTestSplit
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


__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "_checks": ["Check", "CheckNotApplicable", "ChecksSummaryDisplay"],
        "_config": ["configuration"],
        "_dispatch": ["compare", "evaluate"],
        "_displays": [
            "ConfusionMatrixDisplay",
            "MetricsSummaryDisplay",
            "PrecisionRecallCurveDisplay",
            "PredictionErrorDisplay",
            "RocCurveDisplay",
            "TableReportDisplay",
        ],
        "_displays.base": ["Display"],
        "_displays.inspection.calibration_curve": ["CalibrationDisplay"],
        "_displays.inspection.coefficients": ["CoefficientsDisplay"],
        "_displays.inspection.impurity_decrease": ["ImpurityDecreaseDisplay"],
        "_displays.inspection.permutation_importance": ["PermutationImportanceDisplay"],
        "_project.login": ["login"],
        "_project.project": ["Project"],
        "_project.summary": ["Summary"],
        "_reports": [
            "ComparisonReport",
            "CrossValidationReport",
            "EstimatorReport",
        ],
        "_sklearn": ["TrainTestSplit"],
        "_utils.show_versions": ["show_versions"],
    },
)
