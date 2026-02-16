from importlib.metadata import entry_points
from typing import get_args

from skore._project.types import PluginGroup, ProjectMode

GROUPS = get_args(PluginGroup)
MODES = get_args(ProjectMode)


def get(*, group: PluginGroup, mode: ProjectMode):
    assert group in GROUPS, f"`group` must be in {GROUPS} (found {group})"
    assert mode in MODES, f"`mode` must be in {MODES} (found {mode})"

    plugins = entry_points(group=group)

    if mode not in plugins.names:
        raise ValueError(f"Unknown mode `{mode}`. Please install `skore[{mode}]`.")

    return plugins[mode].load()
