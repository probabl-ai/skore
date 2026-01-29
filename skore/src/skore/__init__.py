"""Configure logging and global settings."""

from importlib.metadata import entry_points
from logging import INFO, NullHandler, getLogger
from typing import Literal
from warnings import warn

from joblib import __version__ as joblib_version
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
from skore._sklearn._plot.feature_importance.coefficients import CoefficientsDisplay
from skore._utils._patch import setup_jupyter_display
from skore._utils._show_versions import show_versions
from skore.project import Project

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
    "MetricsSummaryDisplay",
    "PrecisionRecallCurveDisplay",
    "PredictionErrorDisplay",
    "Project",
    "RocCurveDisplay",
    "TableReportDisplay",
    "config_context",
    "console",
    "get_config",
    "logger",
    "login",
    "set_config",
    "show_versions",
    "skore_console_theme",
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


def login(*, mode: Literal["local", "hub"] = "hub", **kwargs):
    """
    Login to the storage backend.

    It configures the credentials used to communicate with the desired storage backend.
    It only affects the current session: credentials are stored in memory and are not
    persisted.

    It must be called at the top of your script.

    Two storage backends are available and their credentials can be configured using the
    ``mode`` parameter. Please note that both can be configured in the same script
    without side-effect.

    .. rubric:: Hub mode

    It configures the credentials used to communicate with the ``Skore Hub``, persisting
    projects remotely.

    In this mode, you must be already registered to the ``Skore Hub``. Also, we
    recommend that you set up an API key via [url](coming soon) and use it to log in.

    .. rubric:: Local mode

    Otherwise, it configures the credentials used to communicate with the ``local``
    backend, persisting projects on the user machine.

    Parameters
    ----------
    mode : Literal["local", "hub"], optional
        The mode of the storage backend to log in, default hub.
    **kwargs : dict
        Extra keyword arguments passed to the login function, depending on its mode.

        timeout : int, mode:hub only, optional
            The maximum time in second before raising an error, default 600.

    See Also
    --------
    :class:`~skore.Project` :
        Refer to the :ref:`project` section of the user guide for more details.
    """
    if mode == "local":
        return

    MODE = "hub"
    PLUGINS = entry_points(group="skore.plugins.login")

    if MODE not in PLUGINS.names:
        raise ValueError(f"Unknown mode `{MODE}`. Please install `skore[{MODE}]`.")

    return PLUGINS[MODE].load()(**kwargs)
