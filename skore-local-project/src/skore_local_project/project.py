"""Class definition of the ``skore`` local project."""

from __future__ import annotations

import io
import os
from contextlib import suppress
from datetime import datetime, timezone
from operator import itemgetter
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import uuid4

import joblib
import platformdirs

from .storage import DiskCacheStorage

if TYPE_CHECKING:
    from typing import Any, Optional, TypedDict, Union

    from skore import EstimatorReport

    class PersistedMetadata:  # noqa: D101
        artifact_id: str
        project_name: str
        run_id: str
        key: str
        date: str
        learner: str
        dataset: str
        ml_task: str
        rmse: Union[float, None]
        log_loss: Union[float, None]
        roc_auc: Union[float, None]
        fit_time: float
        predict_time: float

    class Metadata(TypedDict):  # noqa: D101
        id: str
        run_id: str
        key: str
        date: str
        learner: str
        dataset: str
        ml_task: str
        rmse: Union[float, None]
        log_loss: Union[float, None]
        roc_auc: Union[float, None]
        fit_time: float
        predict_time: float


def lazy_is_instance_skore_estimator_report(value: Any) -> bool:
    """Return True if value is an instance of ``skore.EstimatorReport``."""
    return "skore.sklearn._estimator.report.EstimatorReport" in {
        f"{cls.__module__}.{cls.__name__}" for cls in value.__class__.__mro__
    }


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
        | The workspace can be set using kwargs or the envar ``SKORE_WORKSPACE``.
        | If not, it will be by default set to a ``skore/`` directory in the USER
        cache directory:

        - in Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
        - in Linux, usually ``${HOME}/.cache/skore``,
        - in macOS, usually ``${HOME}/Library/Caches/skore``.
    run_id : str
        The current run identifier of the project.
    """

    def __init__(self, name: str, *, workspace: Optional[Path] = None):
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
            | The workspace can be set using kwargs or the envar ``SKORE_WORKSPACE``.
            | If not, it will be by default set to a ``skore/`` directory in the USER
            cache directory:

            - in Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
            - in Linux, usually ``${HOME}/.cache/skore``,
            - in macOS, usually ``${HOME}/Library/Caches/skore``.
        """
        if workspace is None:
            if "SKORE_WORKSPACE" in os.environ:
                workspace = Path(os.environ["SKORE_WORKSPACE"])
            else:
                workspace = Path(platformdirs.user_cache_dir()) / "skore"

        (workspace / "metadata").mkdir(parents=True, exist_ok=True)
        (workspace / "artifacts").mkdir(parents=True, exist_ok=True)

        self.workspace = str(workspace)
        self.name = name
        self.run_id = uuid4().hex
        self.__metadata_storage = DiskCacheStorage(workspace / "metadata")
        self.__artifacts_storage = DiskCacheStorage(workspace / "artifacts")

    @staticmethod
    def pickle(report: EstimatorReport) -> tuple[str, bytes]:
        """
        Pickle ``report``, return the bytes and the corresponding hash.

        Notes
        -----
        The report is pickled without its cache, to avoid salting the hash.
        """
        cache = report._cache

        try:
            report._cache = {}

            with io.BytesIO() as stream:
                joblib.dump(report, stream)

                pickle_bytes = stream.getvalue()
                pickle_hash = joblib.hash(pickle_bytes)
        finally:
            report._cache = cache

        return pickle_hash, pickle_bytes

    def put(self, key: str, report: EstimatorReport):
        """
        Put a key-report pair to the local project.

        If the key already exists, its last report is modified to point to this new
        report, while keeping track of the report history.

        Parameters
        ----------
        key : str
            The key to associate with ``report`` in the local project.
        report : skore.EstimatorReport
            The report to associate with ``key`` in the local project.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        if not lazy_is_instance_skore_estimator_report(report):
            raise TypeError(
                f"Report must be a `skore.EstimatorReport` (found '{type(report)}')"
            )

        pickle_hash, pickle_bytes = self.pickle(report)

        if pickle_hash not in self.__artifacts_storage:
            self.__artifacts_storage[pickle_hash] = pickle_bytes

        def metric(name):
            """
            Compute ``report.metrics.name``.

            Notes
            -----
            Unavailable metrics return None.

            All metrics whose report is not a scalar return None:
            - ignore ``list[float]`` for multi-output ML task,
            - ignore ``dict[str: float]`` for multi-classes ML task.
            """
            if hasattr(report.metrics, name):
                with suppress(TypeError):
                    return float(getattr(report.metrics, name)(data_source="test"))
            return None

        self.__metadata_storage[uuid4().hex] = {
            "project_name": self.name,
            "run_id": self.run_id,
            "key": key,
            "artifact_id": pickle_hash,
            "date": datetime.now(timezone.utc).isoformat(),
            "learner": report.estimator_name_,
            "dataset": joblib.hash(report.y_test),
            "ml_task": report._ml_task,
            "rmse": metric("rmse"),
            "log_loss": metric("log_loss"),
            "roc_auc": metric("roc_auc"),
            # timings must be calculated last
            "fit_time": report.metrics.timings().get("fit_time"),
            "predict_time": report.metrics.timings().get("predict_time_test"),
        }

    @property
    def reports(self):
        """Accessor for interaction with the persisted reports."""

        def get(id: str) -> EstimatorReport:
            """Get a persisted report by its id."""
            if id in self.__artifacts_storage:
                with io.BytesIO(self.__artifacts_storage[id]) as stream:
                    return joblib.load(stream)

            raise KeyError(id)

        def metadata() -> list[Metadata]:
            """Obtain metadata for all persisted reports regardless of their run."""
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
        return f"Project(local:{self.workspace}@{self.name})"
