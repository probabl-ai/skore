import io
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from operator import itemgetter
from pathlib import Path
from typing import Optional, Any
from uuid import uuid4

import joblib

from skore import EstimatorReport
from skore.persistence.storage import DiskCacheStorage


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
    def __init__(self, name, *, workspace=(Path.home() / ".cache" / "skore")):
        (workspace / "metadata").mkdir(parents=True)
        (workspace / "artifacts").mkdir(parents=True)

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

        if isinstance(value, EstimatorReport):

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

    # def get(self, key):
    #     metadata = max(
    #         (
    #             metadata
    #             for metadata in self.metadata_storage.values()
    #             if metadata["project_name"] == self.name and metadata["key"] == key
    #         ),
    #         key=itemgetter("date"),
    #         default=None,
    #     )

    #     if metadata:
    #         with io.BytesIO(self.artifacts_storage[metadata["artifact_id"]]) as stream:
    #             return joblib.load(stream)

    #     raise KeyError

    @property
    def experiments(self):
        class Namespace:
            @staticmethod
            def __call__(id: int):
                if id in self.artifacts_storage:
                    with io.BytesIO(self.artifacts_storage[id]) as stream:
                        return joblib.load(stream)

                raise KeyError

            @staticmethod
            def metadata():
                return sorted(
                    (
                        {
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
                        for value in self.metadata_storage.values()
                        if (value["project_name"] == self.name) and value["experiment"]
                    ),
                    key=itemgetter("date"),
                )

        return Namespace()


if __name__ == "__main__":
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from skore import EstimatorReport
    from skore.scratch import Project

    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    classifier = LogisticRegression(max_iter=10_000)
    report = EstimatorReport(
        classifier,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    with TemporaryDirectory() as tmpdir:
        # setup
        project = Project("test", workspace=Path(tmpdir))

        # put
        project.put("int", 0)
        project.put("int", 1)
        project.put("int", 2)
        project.put("float", 0.0)
        project.put("report", report)

        # get
        print(project.get("int"))

        # metadata
        print(project.experiments.metadata())
