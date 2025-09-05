"""Class definition of the ``skore`` local project."""

from __future__ import annotations

import io
import os
from functools import wraps
from operator import itemgetter
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import uuid4

import joblib
import platformdirs
from skore import CrossValidationReport, EstimatorReport

from .metadata import CrossValidationReportMetadata, EstimatorReportMetadata
from .storage import DiskCacheStorage

if TYPE_CHECKING:
    from typing import TypedDict

    class Metadata(TypedDict):  # noqa: D101
        id: str
        run_id: str
        key: str
        date: str
        learner: str
        dataset: str
        ml_task: str
        rmse: float | None
        log_loss: float | None
        roc_auc: float | None
        fit_time: float
        predict_time: float


def ensure_project_is_not_deleted(method):
    """Ensure project is not deleted, before executing any other operation."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.name not in self._Project__projects_storage:
            raise RuntimeError(
                f"Skore could not proceed because {repr(self)} does not exist anymore."
            )

        return method(self, *args, **kwargs)

    return wrapper


class Project:
    r"""
    API to manage a collection of key-report pairs persisted in a local storage.

    It communicates with a ``diskcache`` storage, based on the pickle representation.
    Its constructor initializes a local project by creating a new project or by
    loading an existing one from a ``workspace``.

    The class main methods are :func:`~skore_local_project.Project.put`,
    :func:`~skore_local_project.reports.metadata` and
    :func:`~skore_local_project.reports.get`, respectively to insert a key-report pair
    into the Project, to obtain the reports metadata and to get a specific report.

    Attributes
    ----------
    name : str
        The name of the project.
    workspace : Path
        The directory where the project (metadata and artifacts) are persisted.

        | The workspace can be shared between all the projects.
        | The workspace can be set using kwargs or the environment variable
          ``SKORE_WORKSPACE``.
        | If not, it will be by default set to a ``skore/`` directory in the USER
          cache directory:

        - on Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
        - on Linux, usually ``${HOME}/.cache/skore``,
        - on macOS, usually ``${HOME}/Library/Caches/skore``.
    run_id : str
        The current run identifier of the project.
    """

    @staticmethod
    def __setup_diskcache(
        workspace: Path | None,
    ) -> tuple[
        Path,
        DiskCacheStorage,
        DiskCacheStorage,
        DiskCacheStorage,
    ]:
        if workspace is None:
            if "SKORE_WORKSPACE" in os.environ:
                workspace = Path(os.environ["SKORE_WORKSPACE"])
            else:
                workspace = Path(platformdirs.user_cache_dir()) / "skore"

        for directory in ("projects", "metadata", "artifacts"):
            (workspace / directory).mkdir(parents=True, exist_ok=True)

        return (
            workspace,
            DiskCacheStorage(workspace / "projects"),
            DiskCacheStorage(workspace / "metadata"),
            DiskCacheStorage(workspace / "artifacts"),
        )

    def __init__(self, name: str, *, workspace: Path | None = None):
        r"""
        Initialize a local project.

        Initialize a local project by creating a new project or by loading an existing
        one from the ``workspace``.

        Parameters
        ----------
        name : str
            The name of the project.
        workspace : Path, optional
            The directory where the project (metadata and artifacts) are persisted.

            | The workspace can be shared between all the projects.
            | The workspace can be set using kwargs or the environment variable
              ``SKORE_WORKSPACE``.
            | If not, it will be by default set to a ``skore/`` directory in the USER
              cache directory:

            - on Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
            - on Linux, usually ``${HOME}/.cache/skore``,
            - on macOS, usually ``${HOME}/Library/Caches/skore``.
        """
        workspace, projects, metadata, artifacts = Project.__setup_diskcache(workspace)

        self.__name = name
        self.__run_id = uuid4().hex
        self.__workspace = workspace
        self.__projects_storage = projects
        self.__metadata_storage = metadata
        self.__artifacts_storage = artifacts

        # Create the project
        self.__projects_storage[name] = None

    @property
    def name(self) -> str:
        """The name of the project."""
        return self.__name

    @property
    def run_id(self) -> str:
        """The run identifier of the project."""
        return self.__run_id

    @property
    def workspace(self) -> Path:
        """The workspace of the project."""
        return self.__workspace

    @staticmethod
    def pickle(report: EstimatorReport | CrossValidationReport) -> tuple[str, bytes]:
        """
        Pickle ``report``, return the bytes and the corresponding hash.

        Notes
        -----
        The report is pickled without its cache, to avoid salting the hash.
        """
        reports = [report] + getattr(report, "estimator_reports_", [])
        caches = [report_to_clear._cache for report_to_clear in reports]

        report.clear_cache()

        try:
            with io.BytesIO() as stream:
                joblib.dump(report, stream)

                pickle_bytes = stream.getvalue()
                pickle_hash = joblib.hash(pickle_bytes)
        finally:
            for report, cache in zip(reports, caches, strict=True):
                report._cache = cache

        return pickle_hash, pickle_bytes

    @ensure_project_is_not_deleted
    def put(self, key: str, report: EstimatorReport | CrossValidationReport):
        """
        Put a key-report pair to the local project.

        If the key already exists, its last report is modified to point to this new
        report, while keeping track of the report history.

        Parameters
        ----------
        key : str
            The key to associate with ``report`` in the local project.
        report : skore.EstimatorReport | skore.CrossValidationReport
            The report to associate with ``key`` in the local project.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        Metadata: type[EstimatorReportMetadata] | type[CrossValidationReportMetadata]

        if isinstance(report, EstimatorReport):
            Metadata = EstimatorReportMetadata
        elif isinstance(report, CrossValidationReport):
            Metadata = CrossValidationReportMetadata
        else:
            raise TypeError(
                f"Report must be a `skore.EstimatorReport` or `skore.CrossValidationRep"
                f"ort` (found '{type(report)}')"
            )

        pickle_hash, pickle_bytes = Project.pickle(report)

        if pickle_hash not in self.__artifacts_storage:
            self.__artifacts_storage[pickle_hash] = pickle_bytes

        self.__metadata_storage[uuid4().hex] = dict(
            Metadata(
                report=report,
                artifact_id=pickle_hash,
                project_name=self.name,
                run_id=self.run_id,
                key=key,
            )
        )

    @property
    @ensure_project_is_not_deleted
    def reports(self):
        """Accessor for interaction with the persisted reports."""

        def get(id: str) -> EstimatorReport:
            """Get a persisted report by its id."""
            if id in self.__artifacts_storage:
                with io.BytesIO(self.__artifacts_storage[id]) as stream:
                    return joblib.load(stream)

            raise KeyError(id)

        def metadata() -> list[Metadata]:
            """Obtain metadata/metrics for all persisted reports."""
            return sorted(
                (
                    {
                        "id": value["artifact_id"],
                        "run_id": value["run_id"],
                        "key": value["key"],
                        "date": value["date"],
                        "learner": value["learner"],
                        "dataset": value["dataset"],
                        "ml_task": value["ml_task"],
                        "rmse": value["rmse"],
                        "log_loss": value["log_loss"],
                        "roc_auc": value["roc_auc"],
                        "fit_time": value["fit_time"],
                        "predict_time": value["predict_time"],
                    }
                    for value in self.__metadata_storage.values()
                    if value["project_name"] == self.name
                ),
                key=itemgetter("date"),
            )

        return SimpleNamespace(get=get, metadata=metadata)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"Project(mode='local', name='{self.name}', workspace='{self.workspace}')"
        )

    @staticmethod
    def delete(name: str, *, workspace: Path | None = None):
        r"""
        Delete a local project.

        Parameters
        ----------
        name : str
            The name of the project.
        workspace : Path, optional
            The directory where the project (metadata and artifacts) are persisted.

            | The workspace can be shared between all the projects.
            | The workspace can be set using kwargs or the environment variable
              ``SKORE_WORKSPACE``.
            | If not, it will be by default set to a ``skore/`` directory in the USER
              cache directory:

            - on Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
            - on Linux, usually ``${HOME}/.cache/skore``,
            - on macOS, usually ``${HOME}/Library/Caches/skore``.
        """
        workspace, projects, metadata, artifacts = Project.__setup_diskcache(workspace)

        if name not in projects:
            raise LookupError(
                f"Project(mode='local', name='{name}', workspace='{workspace}') "
                f"does not exist."
            )

        # Delete all metadata related to the project
        remaining_artifacts = set()

        for key, value in metadata.items():
            if value["project_name"] == name:
                del metadata[key]
            else:
                remaining_artifacts.add(value["artifact_id"])

        # Prune artifacts not related to a project
        for artifact in artifacts:
            if artifact not in remaining_artifacts:
                del artifacts[artifact]

        # Delete the project
        del projects[name]
