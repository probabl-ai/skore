"""Class definition of the ``skore`` local project."""

from __future__ import annotations

import io
import os
from contextlib import suppress
from dataclasses import asdict, dataclass
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

    class EstimatorReportMetadata(TypedDict):  # noqa: D101
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


def lazy_is_instance(value: Any, cls_fullname: str) -> bool:
    """Return True if value is an instance of ``cls_fullname``."""
    return cls_fullname in {
        f"{cls.__module__}.{cls.__name__}" for cls in value.__class__.__mro__
    }


@dataclass
class Metadata:
    """Definition of the metadata that are persisted."""

    id: str
    project_name: str
    run_id: str
    key: str
    artifact_id: str
    date: str
    note: Optional[str] = None
    experiment: Optional[bool] = False
    various: Optional[dict] = None


class Project:
    r"""
    API to manage a collection of key-value pairs persisted in a local storage.

    It communicates with a ``diskcache`` storage.
    Its constructor initializes a local project by creating a new project or by
    loading an existing one from a ``workspace``.

    The class main methods are :func:`~skore.Project.put`,
    :func:`~skore.Project.experiments.metadata` and
    :func:`~skore.Project.experiments.get`, respectively to insert a key-value pair into
    the Project, to obtain the experiments metadata and to get a specific experiment.

    You can add any type of objects. In some cases, especially on classes you defined,
    the persistency is based on the pickle representation.

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

            - in Windows, usually ``C:\Users\%USER%\AppData\Local``,
            - in Linux, usually ``${HOME}/.cache``,
            - in macOS, usually ``${HOME}/Library/Caches``.
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

            - in Windows, usually ``C:\Users\%USER%\AppData\Local``,
            - in Linux, usually ``${HOME}/.cache``,
            - in macOS, usually ``${HOME}/Library/Caches``.
        """
        if workspace is None:
            if "SKORE_WORKSPACE" in os.environ:
                workspace = Path(os.environ["SKORE_WORKSPACE"]) / "skore"
            else:
                workspace = Path(platformdirs.user_cache_dir()) / "skore"

        (workspace / "metadata").mkdir(parents=True, exist_ok=True)
        (workspace / "artifacts").mkdir(parents=True, exist_ok=True)

        self.workspace = str(workspace)
        self.name = name
        self.run_id = uuid4().hex
        self.metadata_storage = DiskCacheStorage(workspace / "metadata")
        self.artifacts_storage = DiskCacheStorage(workspace / "artifacts")

    def put(self, key: str, value: Any, *, note: Optional[str] = None):
        """
        Put a key-value pair to the local project.

        If the key already exists, its last value is modified to point to this new
        value, while keeping track of the value history.

        Parameters
        ----------
        key : str
            The key to associate with ``value`` in the local project.
        value : Any
            The value to associate with ``key`` in the local project.
        note : str, optional
            A note to attach with the key-value pair, default None.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        if not isinstance(note, (type(None), str)):
            raise TypeError(f"Note must be a string (found '{type(note)}')")

        id = uuid4().hex
        now = datetime.now(timezone.utc).isoformat()

        with io.BytesIO() as stream:
            joblib.dump(value, stream)

            pickle_bytes = stream.getvalue()
            pickle_hash = joblib.hash(pickle_bytes)

        if pickle_hash not in self.artifacts_storage:
            self.artifacts_storage[pickle_hash] = pickle_bytes

        if lazy_is_instance(value, "skore.sklearn._estimator.report.EstimatorReport"):

            def metric(name):
                """
                Compute ``report.metrics.name``.

                Notes
                -----
                Unavailable metrics return None.

                All metrics whose value is not a scalar return None:
                - ignore ``list[float]`` for multi-output ML task,
                - ignore ``dict[str: float]`` for multi-classes ML task.
                """
                if hasattr(value.metrics, name):
                    with suppress(TypeError):
                        return float(getattr(value.metrics, name)(data_source="test"))
                return None

            self.metadata_storage[id] = asdict(
                Metadata(
                    id=id,
                    project_name=self.name,
                    run_id=self.run_id,
                    key=key,
                    artifact_id=pickle_hash,
                    date=now,
                    note=note,
                    experiment=True,
                    various={
                        "learner": value.estimator_name_,
                        "dataset": joblib.hash(value.y_test),
                        "ml_task": value._ml_task,
                        "rmse": metric("rmse"),
                        "log_loss": metric("log_loss"),
                        "roc_auc": metric("roc_auc"),
                        # timings must be calculated last
                        "fit_time": value.metrics.timings().get("fit_time"),
                        "predict_time": value.metrics.timings().get(
                            "predict_time_test"
                        ),
                    },
                )
            )
        else:
            self.metadata_storage[id] = asdict(
                Metadata(
                    id=id,
                    project_name=self.name,
                    run_id=self.run_id,
                    key=key,
                    artifact_id=pickle_hash,
                    date=now,
                    note=note,
                )
            )

    @property
    def experiments(self):
        """Accessor for interaction with the persisted experiments."""

        def get(id: str) -> EstimatorReport:
            """Get a persisted experiment by its id."""
            if id in self.artifacts_storage:
                with io.BytesIO(self.artifacts_storage[id]) as stream:
                    return joblib.load(stream)

            raise KeyError

        @staticmethod
        def metadata() -> list[EstimatorReportMetadata]:
            """
            Obtain metadata for all persisted experiments regardless of their run.

            Notes
            -----
            Only scalar metrics are listed in the metadata.
            """
            return sorted(
                (
                    {
                        "id": value["artifact_id"],
                        "run_id": value["run_id"],
                        "key": value["key"],
                        "date": value["date"],
                        "learner": value["various"]["learner"],
                        "dataset": value["various"]["dataset"],
                        "ml_task": value["various"]["ml_task"],
                        "rmse": value["various"]["rmse"],
                        "log_loss": value["various"]["log_loss"],
                        "roc_auc": value["various"]["roc_auc"],
                        "fit_time": value["various"]["fit_time"],
                        "predict_time": value["various"]["predict_time"],
                    }
                    for value in self.metadata_storage.values()
                    if (value["project_name"] == self.name) and value["experiment"]
                ),
                key=itemgetter("date"),
            )

        return SimpleNamespace(get=get, metadata=metadata)

    def __repr__(self) -> str:  # noqa: D105
        return f"Project(local:{self.workspace}@{self.name})"
