"""Configure storage backend credentials."""

from logging import getLogger

from skore._project import plugin
from skore._project.dependencies import assert_optional_dependencies_installed
from skore._project.types import ProjectMode

logger = getLogger(__name__)


def login(*, mode: ProjectMode = "hub", **kwargs):
    """
    Log in to Skore Hub for the duration of the session (e.g. script).

    This command is only useful if you have an account on Skore Hub and wish
    to push artifacts to it.

    By default, it will open a login screen on your browser. However, this login only
    persists for the lifetime of the Python process (e.g. one run of a script, or one
    Jupyter session), so you will have to authenticate via your browser at every run.
    The recommended way to connect to Skore Hub for repeated script runs is using an
    API key; refer to the Skore Hub documentation for how to create one.

    Parameters
    ----------
    mode : {"hub", "local", "mlflow"}, default="hub"
        The mode of the storage backend to log in. If the mode is not "hub", the
        function is a no-op.

    **kwargs : dict
        Extra keyword arguments passed to the login function, depending on its mode.

        Arguments for ``mode="hub"``:

        timeout : int, default=600
            The time, in seconds, before raising an error if communication with
            Skore Hub fails.

    Returns
    -------
    None
        For ``mode="local"`` and ``mode="mlflow"``. For ``mode="hub"``, the return
        value depends on the hub login plugin.

    Examples
    --------
    >>> from skore import login
    >>> login(mode="local")

    See Also
    --------
    :class:`~skore.Project` :
        Refer to the :ref:`project` section of the user guide for more details.
    """
    if mode not in {"hub", "local", "mlflow"}:
        raise ValueError(f'`mode` must be "hub", "local" or "mlflow" (found {mode})')

    assert_optional_dependencies_installed(mode)

    if mode == "local":
        logger.debug("Login to local storage.")
        return

    if mode == "mlflow":
        logger.debug("Login to MLflow storage.")
        return

    logger.debug("Login to hub storage.")

    return plugin.get(group="skore.plugins.login", mode="hub")(**kwargs)
