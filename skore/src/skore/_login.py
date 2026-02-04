"""Configure storage backend credentials."""

from importlib.metadata import entry_points
from logging import getLogger
from typing import Literal

logger = getLogger(__name__)


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
    mode : {"local", "hub"}, default="hub"
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

    mode = "hub"
    plugins = entry_points(group="skore.plugins.login")

    if mode not in plugins.names:
        raise ValueError(f"Unknown mode `{mode}`. Please install `skore[{mode}]`.")

    logger.debug("Login to hub storage.")

    return plugins[mode].load()(**kwargs)
