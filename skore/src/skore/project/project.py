from __future__ import annotations

import re
import sys
import types

import pandas as pd

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from ..sklearn._estimator.report import EstimatorReport
from .metadata import Metadata


class Project:
    __REMOTE_NAME_PATTERN = re.compile(r"remote://(?P<tenant>[^/]+)/(?P<name>.+)")

    def __init__(self, name: str, **kwargs):
        if not (PLUGINS := entry_points(group="skore.plugins.project")):
            raise SystemError("No project plugin found, please install at least one.")

        if match := re.match(self.__REMOTE_NAME_PATTERN, name):
            mode = "remote"
            kwargs |= {"tenant": match["tenant"], "name": match["name"]}
        else:
            mode = "local"
            kwargs |= {"name": name}

        if mode not in PLUGINS.names:
            raise ValueError(
                f"Unknown mode '{mode}'. Available modes {', '.join(PLUGINS.names)}."
            )

        self.__mode = mode
        self.__name = name
        self.__project = PLUGINS[mode].load()(**kwargs)

    def put(self, key: str, report: EstimatorReport):
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        if not isinstance(report, EstimatorReport):
            raise TypeError(
                f"Report must be a `skore.EstimatorReport` (found '{type(report)}')"
            )

        return self.__project.put(key=key, report=report)

    @property
    def reports(self):
        """"""

        def get(id: str) -> EstimatorReport:  # hide underlying functions from user
            """"""
            return self.__project.reports.get(id)

        def metadata() -> Metadata:  # hide underlying functions from user
            """"""
            metadata = pd.DataFrame(self.__project.reports.metadata(), copy=False)
            if not metadata.empty:
                metadata["learner"] = pd.Categorical(metadata["learner"])
                metadata.index = pd.MultiIndex.from_arrays(
                    [
                        pd.RangeIndex(len(metadata)),
                        pd.Index(metadata.pop("id"), name="id", dtype=str),
                    ]
                )

            # Cast standard dataframe to Metadata for lazy reports selection.
            metadata = Metadata(metadata, copy=False)
            metadata.project = self.__project

            return metadata

        return types.SimpleNamespace(get=get, metadata=metadata)

    def __repr__(self) -> str:  # noqa: D105
        return self.__project.__repr__()
