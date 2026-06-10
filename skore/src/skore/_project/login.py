"""Configure storage backend credentials."""

from logging import getLogger

from skore._plugins.hub.authentication.login import login as hub_login
from skore._project.dependencies import assert_optional_dependencies_installed

logger = getLogger(__name__)


def login(*, timeout: int = 600):
    """
    Log in to Skore Hub for the duration of the session (e.g. script).

    This command is only useful if you have an account on Skore Hub and wish
    to push artifacts to it. 

    By default, will open a login screen on your browser. However, this login only
    persists for the lifetime of the Python process (e.g. one run of a script, or one
    Jupyter session).

    The recommended way to connect to Skore Hub for repeated script runs is using an
    API key; refer to the Skore Hub documentation for how to create one.

    Parameters
    ----------
    timeout : int, default=600
        The time, in seconds, before raising an error if communication with Skore Hub
        fails.

    Returns
    -------
    None

    Examples
    --------
    >>> from skore import login
    >>> login()

    See Also
    --------
    :class:`~skore.Project` :
        Refer to the :ref:`project` section of the user guide for more details.
    """
    assert_optional_dependencies_installed("hub")

    logger.debug("Logging in to hub storage.")

    return hub_login(timeout=timeout)
