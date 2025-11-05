"""Class definition of the ``skore`` local project."""

from __future__ import annotations

import io
import os
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, cast, runtime_checkable
from uuid import uuid4

import joblib
import platformdirs

from .metadata import CrossValidationReportMetadata, EstimatorReportMetadata
from .storage import DiskCacheStorage

P = ParamSpec("P")
R = TypeVar("R")


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypedDict

    from skore import CrossValidationReport, EstimatorReport

    class Metadata(TypedDict):  # noqa: D101
        id: str
        key: str
        date: str
        learner: str
        ml_task: str
        report_type: str
        dataset: str
        rmse: float | None
        log_loss: float | None
        roc_auc: float | None
        fit_time: float | None
        predict_time: float | None
        rmse_mean: float | None
        log_loss_mean: float | None
        roc_auc_mean: float | None
        fit_time_mean: float | None
        predict_time_mean: float | None


def ensure_project_is_not_deleted(method: Callable[P, R]) -> Callable[P, R]:
    """Ensure project is not deleted, before executing any other operation."""

    @runtime_checkable
    class Project(Protocol):
        name: str
        _Project__projects_storage: DiskCacheStorage

    @wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        project = args[0]

        assert isinstance(project, Project), "You can only wrap `Project` methods"

        if project.name not in project._Project__projects_storage:
            raise RuntimeError(
                f"Skore could not proceed because {project!r} does not exist anymore."
            )

        return method(*args, **kwargs)

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
        | If not, it will be by default set to a ``skore/`` directory in the user
          cache directory:

        - on Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
        - on Linux, usually ``${HOME}/.local/share/skore``,
        - on macOS, usually ``${HOME}/Library/Application Support/skore``.
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
                workspace = Path(platformdirs.user_data_dir()) / "skore"

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
    def put(self, key: str, report: EstimatorReport | CrossValidationReport) -> None:
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
        from skore import CrossValidationReport, EstimatorReport

        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        Metadata: type[EstimatorReportMetadata | CrossValidationReportMetadata]

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
                key=key,
            )
        )

    @ensure_project_is_not_deleted
    def get(self, id: str) -> EstimatorReport | CrossValidationReport:
        """Get a persisted report by its id."""
        if id in self.__artifacts_storage:
            with io.BytesIO(self.__artifacts_storage[id]) as stream:
                return cast(
                    "EstimatorReport | CrossValidationReport", joblib.load(stream)
                )

        raise KeyError(id)

    @ensure_project_is_not_deleted
    def summarize(self) -> list[Metadata]:
        """Obtain metadata/metrics for all persisted reports in insertion order."""
        return [
            {
                "id": value["artifact_id"],
                "key": value["key"],
                "date": value["date"],
                "learner": value["learner"],
                "ml_task": value["ml_task"],
                "report_type": value["report_type"],
                "dataset": value["dataset"],
                "rmse": value.get("rmse"),
                "log_loss": value.get("log_loss"),
                "roc_auc": value.get("roc_auc"),
                "fit_time": value.get("fit_time"),
                "predict_time": value.get("predict_time"),
                "rmse_mean": value.get("rmse_mean"),
                "log_loss_mean": value.get("log_loss_mean"),
                "roc_auc_mean": value.get("roc_auc_mean"),
                "fit_time_mean": value.get("fit_time_mean"),
                "predict_time_mean": value.get("predict_time_mean"),
            }
            for value in self.__metadata_storage.values()
            if value["project_name"] == self.name
        ]

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"Project(mode='local', name='{self.name}', workspace='{self.workspace}')"
        )

    @staticmethod
    def delete(name: str, *, workspace: Path | None = None) -> None:
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
