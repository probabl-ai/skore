"""Evaluate and compare scikit-learn compatible models with rich reports.

This package provides tools to evaluate estimators, compare models, persist
experiment results, and inspect model behavior through interactive reports.
"""

from importlib.metadata import version
from logging import INFO, NullHandler, getLogger
from warnings import warn

from packaging.version import Version
from rich.console import Console
from rich.theme import Theme

from skore._config import configuration
from skore._externals import _lazy_loader as lazy
from skore._utils._environment import is_environment_notebook_like
from skore._utils._patch import setup_jupyter_display

setup_jupyter_display()


if Version(version("joblib")) < Version("1.4"):
    configuration.show_progress = False
    warn(
        "Because your version of joblib is older than 1.4, some of the features of "
        "skore will not be enabled (e.g. progress bars). You can update joblib to "
        "benefit from these features.",
        stacklevel=2,
    )


__version__ = version("skore")
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)


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
