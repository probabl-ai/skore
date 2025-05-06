from __future__ import annotations

import io
import os
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from operator import itemgetter
from pathlib import Path
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
        note: Union[str, None]
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
    def __init__(self, name: str, *, workspace: Optional[Path] = None):
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

    @staticmethod
    def pickle(value):
        with io.BytesIO() as stream:
            joblib.dump(value, stream)

            pickle_bytes = stream.getvalue()
            pickle_hash = joblib.hash(pickle_bytes)

        return pickle_hash, pickle_bytes

    def put(self, key: str, value: Any, *, note: Optional[str] = None):
        id = uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        pickle_hash, pickle_bytes = self.pickle(value)

        if pickle_hash not in self.artifacts_storage:
            self.artifacts_storage[pickle_hash] = pickle_bytes

        if lazy_is_instance(value, "skore.sklearn._estimator.report.EstimatorReport"):

            def metric(name):
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
        class Namespace:
            @staticmethod
            def __call__(id: int) -> EstimatorReport:
                if id in self.artifacts_storage:
                    with io.BytesIO(self.artifacts_storage[id]) as stream:
                        return joblib.load(stream)

                raise KeyError

            @staticmethod
            def metadata() -> list[EstimatorReportMetadata]:
                def dto(value):
                    return {
                        "id": value["artifact_id"],
                        "run_id": value["run_id"],
                        "key": value["key"],
                        "date": value["date"],
                        "note": value["note"],
                        "learner": value["various"]["learner"],
                        "dataset": value["various"]["dataset"],
                        "ml_task": value["various"]["ml_task"],
                        "rmse": value["various"]["rmse"],
                        "log_loss": value["various"]["log_loss"],
                        "roc_auc": value["various"]["roc_auc"],
                        "fit_time": value["various"]["fit_time"],
                        "predict_time": value["various"]["predict_time"],
                    }

                return sorted(
                    map(
                        dto,
                        (
                            value
                            for value in self.metadata_storage.values()
                            if (value["project_name"] == self.name)
                            and value["experiment"]
                        ),
                    ),
                    key=itemgetter("date"),
                )

        return Namespace()

    def __repr__(self) -> str:
        return f"Project(local://{self.workspace}@{self.name})"
