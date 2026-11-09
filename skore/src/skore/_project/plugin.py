"""Tools used to interact with ``skore`` plugin."""

from importlib.metadata import entry_points
from typing import Any, get_args

from skore._project.types import PluginGroup, ProjectMode

GROUPS = get_args(PluginGroup)
MODES = get_args(ProjectMode)


def get(*, group: PluginGroup, mode: ProjectMode) -> Any:
    """
    Load and return a ``skore`` plugin implementation for the given group and mode.

    Plugins currently allowed are the classes implementing the ``Project`` API,
    registered under the ``skore.plugins.project`` group.

    This function uses internally the python entry points mechanism: each package
    compatible with ``skore`` could expose its own plugins, as long as they are
    registered in the right groups and comply with APIs.

    Parameters
    ----------
    group : PluginGroup
        The group of plugin to search for. Must be:
        - "skore.plugins.project"

    mode : ProjectMode
        The project mode used to select the plugin implementation.
        Must be one of:
        - "hub"
        - "local"
        - "mlflow"

    Returns
    -------
    Any
        The loaded plugin class corresponding to the given group and mode.
    """
    assert group in GROUPS, f"`group` must be in {GROUPS} (found {group})"
    assert mode in MODES, f"`mode` must be in {MODES} (found {mode})"

    return entry_points(group=group)[mode].load()
