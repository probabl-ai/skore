"""Define a Project."""

from __future__ import annotations

import re
import sys
from functools import partial
from types import SimpleNamespace
from typing import Any, Optional

from .metadata import Metadata

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


class Project:
    NAME_PATTERN = re.compile(r"(?P<scheme>[^:]+)://(?P<tenant>[^/]+)/(?P<name>.+)")

    def __init__(self, name: str):
        if not (PLUGINS := entry_points(group="skore.plugins.project")):
            raise SystemError("No project plugin found, please install at least one.")

        if (match := re.match(self.NAME_PATTERN, name)) and (match["scheme"] == "skh"):
            mode, kwargs = "remote", dict(tenant=match["tenant"], name=match["name"])
        else:
            mode, kwargs = "local", dict(name=name)

        if mode not in PLUGINS.names:
            raise ValueError(
                f"Unknown mode '{mode}'. Available modes {', '.join(PLUGINS.names)}."
            )

        self.__mode = mode
        self.__name = name
        self.__project = PLUGINS[mode].load()(**kwargs)

    def put(self, key: str, value: Any, *, note: Optional[str] = None):
        return self.__project.put(key=key, value=value, note=note)

    @property
    def experiments(self):
        return SimpleNamespace(metadata=partial(Metadata.factory, self.__project))

    def __repr__(self) -> str:
        return self.__project.__repr__()
