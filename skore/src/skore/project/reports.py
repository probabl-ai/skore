"""Module to manage the persisted reports."""

from skore.externals._pandas_accessors import DirNamesMixin
from skore.project import Project
from skore.project.metadata import Metadata
from skore.sklearn import EstimatorReport


class _ReportsAccessor(DirNamesMixin):
    """Accessor for interaction with the persisted reports."""

    def __init__(self, project: Project) -> None:
        self.__project = project

    def get(self, id: str) -> EstimatorReport:  # hide underlying functions from user
        """Get a persisted report by its id."""
        return self.__project.reports.get(id)

    def metadata(self) -> Metadata:  # hide underlying functions from user
        """Obtain metadata/metrics for all persisted reports."""
        return Metadata.factory(self.__project)
