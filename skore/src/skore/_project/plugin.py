"""Tools used to interact with ``skore`` plugin."""

from importlib.metadata import entry_points
from typing import Any, get_args

from skore._project.types import PluginGroup, ProjectMode

GROUPS = get_args(PluginGroup)
MODES = get_args(ProjectMode)


def get(*, group: PluginGroup, mode: ProjectMode) -> Any:
    """
    Load and return a ``skore`` plugin implementation for the given group and mode.

    There are currently two types of plugins allowed:
    - the classes implementing the ``Project`` API, registered under the
      ``skore.plugins.project`` group,
    - the functions used to login, registered under the ``skore.plugins.login`` group.

    This function uses internally the python entry points mechanism: each package
    compatible with ``skore`` could expose its own plugins, as long as they are
    registered in the right groups and comply with APIs.

    It is usually used to retrieve the plugins exposed by ``skore-local-project``
    or ``skore-hub-project``.

    Parameters
    ----------
    group : PluginGroup
        The group of plugin to search for. Must be one of:
        - "skore.plugins.project"
        - "skore.plugins.login"

    mode : ProjectMode
        The project mode used to select the plugin implementation.
        Must be one of:
        - "hub"
        - "local"

    Returns
    -------
    Any
        The loaded plugin object corresponding to the given group and mode.
        The exact return type depends on the registered plugin implementation: class or
        function.
    """
    assert group in GROUPS, f"`group` must be in {GROUPS} (found {group})"
    assert mode in MODES, f"`mode` must be in {MODES} (found {mode})"

    plugins = entry_points(group=group)

    if mode not in plugins.names:
        raise ValueError(
            f"The mode `{mode}` is not supported. You need to install "
            f"`skore-{mode}-project` to use it. You can install it with pip:\n"
            f'    pip install "skore[{mode}]"\n'
            f"`skore-{mode}-project` is already included in `skore` conda package."
        )

    return plugins[mode].load()
