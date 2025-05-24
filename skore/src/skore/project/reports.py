"""Module to manage the persisted reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skore.externals._pandas_accessors import DirNamesMixin
from skore.project.metadata import Metadata

if TYPE_CHECKING:
    from skore.project import Project
    from skore.sklearn import EstimatorReport


class _ReportsAccessor(DirNamesMixin):
    """Accessor for interaction with the persisted reports."""

    def __init__(self, parent: Project) -> None:
        self._parent = parent

    def get(self, id: str) -> EstimatorReport:  # hide underlying functions from user
        """Get a persisted report by its id."""
        return self._parent._Project__project.reports.get(id)

    def metadata(self) -> Metadata:  # hide underlying functions from user
        """Obtain metadata/metrics for all persisted reports."""
        return Metadata.factory(self._parent._Project__project)
