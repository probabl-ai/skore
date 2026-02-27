"""Configure storage backend credentials."""

from logging import getLogger

from skore._project import plugin
from skore._project.types import ProjectMode

logger = getLogger(__name__)


def login(*, mode: ProjectMode = "hub", **kwargs):
    """
    Login to the storage backend.

    It configures the credentials used to communicate with the desired storage backend.
    It only affects the current session: credentials are stored in memory and are not
    persisted.

    It must be called at the top of your script.

    Several storage backends are available and their credentials can be configured
    using the ``mode`` parameter. Please note that both can be configured in the
    same script without side-effect.

    .. rubric:: Hub mode

    It configures the credentials used to communicate with the ``Skore Hub``, persisting
    projects remotely.

    In this mode, you must be already registered to the ``Skore Hub``. Also, we
    recommend that you set up an API key via https://skore.probabl.ai/account
    and use it to log in.

    .. rubric:: Local mode

    Otherwise, it configures the credentials used to communicate with the ``local``
    backend, persisting projects on the user machine.

    .. rubric:: MLflow mode

    For MLflow projects, no explicit authentication step is required by ``skore`` and
    this function is a no-op.

    Parameters
    ----------
    mode : {"local", "hub", "mlflow"}, default="hub"
        The mode of the storage backend to log in.
    **kwargs : dict
        Extra keyword arguments passed to the login function, depending on its mode.

        timeout : int, default=600
            The maximum time in second before raising an error when communicating with
            the hub. Only available when `mode="hub"`.

    See Also
    --------
    :class:`~skore.Project` :
        Refer to the :ref:`project` section of the user guide for more details.
    """
    if mode == "local":
        logger.debug("Login to local storage.")
        return

    if mode == "mlflow":
        logger.debug("Login to MLflow storage.")
        return

    logger.debug("Login to hub storage.")

    return plugin.get(group="skore.plugins.login", mode="hub")(**kwargs)
