"""Class definition of the ``skore`` MLflow project."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict, cast

import mlflow
import mlflow.sklearn
from joblib import dump as joblib_dump
from joblib import hash
from joblib import load as joblib_load
from mlflow.exceptions import MlflowException
from mlflow.utils.autologging_utils import disable_discrete_autologging

from .metrics import Artifact, Metric
from .protocol import CrossValidationReport, EstimatorReport
from .reports import Params, RawDataArtifact, Tag, iter_cv, iter_estimator

HTML_UTF8_TEMPLATE = """
<head>
<meta charset="UTF-8">
</head>
<body>
{html}
</body>
"""


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


def report_type(report: EstimatorReport | CrossValidationReport) -> str:
    """Human readable type of a report."""
    if isinstance(report, CrossValidationReport):
        return "cross-validation"
    if isinstance(report, EstimatorReport):
        return "estimator"

    raise TypeError(
        f"Report must be a `skore.EstimatorReport` or "
        f"`skore.CrossValidationReport` (found '{type(report)}')."
    )


def format_date(start_time: int | None) -> str:
    """Convert a MLflow run start time (milliseconds) to an ISO datetime."""
    if start_time is None:
        return ""

    return datetime.fromtimestamp(start_time / 1_000, tz=UTC).isoformat()


def _wrap_html(html: str) -> str:
    return HTML_UTF8_TEMPLATE.format(html=html)


def clean_df(df: Any) -> Any:
    """Normalize a dataframe-like object before CSV logging."""
    if not callable(getattr(df, "copy", None)):
        return df

    df = df.copy(deep=False)
    columns = getattr(df, "columns", None)
    if (
        columns is not None
        and getattr(columns, "nlevels", 1) > 1
        and callable(getattr(columns, "droplevel", None))
    ):
        df.columns = columns.droplevel(0)

    index = getattr(df, "index", None)
    is_range_index = index is not None and type(index).__name__ == "RangeIndex"
    if is_range_index and len(getattr(index, "names", [])) == 1:
        return df

    if callable(getattr(df, "reset_index", None)):
        return df.reset_index()
    return df


def log_artifact(artifact: Artifact) -> None:
    """Log a report artifact."""

    def filename(ext: str) -> str:
        return f"report/{artifact.name}.{ext}"

    payload = artifact.payload
    if callable(getattr(payload, "to_csv", None)):
        csv_text = clean_df(payload).to_csv(index=False)
        mlflow.log_text(csv_text, filename("csv"))
    elif callable(getattr(payload, "savefig", None)):
        mlflow.log_figure(payload, filename("png"))
    elif isinstance(payload, list):
        mlflow.log_dict({"values": payload}, filename("json"))
    elif isinstance(payload, dict):
        mlflow.log_dict(payload, filename("json"))
    elif isinstance(payload, str):
        html_text = _wrap_html(payload)
        mlflow.log_text(html_text, filename("html"))
    else:
        raise TypeError(f"Unexpected artifact payload type: {type(payload)}")


class Project:
    """
    API to persist in MLflow.

    It communicates with an MLflow tracking server and stores models associated with
    ``skore`` reports.

    Parameters
    ----------
    name : str
        The name of the MLflow experiment.
    tracking_uri : str, optional
        URI of the MLflow tracking server. If ``None``, MLflow default behavior applies.
    """

    def __init__(self, name: str, *, tracking_uri: str | None = None):
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        self.__tracking_uri = mlflow.get_tracking_uri()
        self.__name = name

        experiment = mlflow.get_experiment_by_name(name)
        self.__experiment_id = (
            cast(str, experiment.experiment_id)
            if experiment is not None
            else mlflow.create_experiment(name)
        )

    @property
    def name(self) -> str:
        """The name of the MLflow experiment."""
        return self.__name

    @property
    def tracking_uri(self) -> str:
        """The URI of the MLflow tracking server."""
        return self.__tracking_uri

    @property
    def experiment_id(self) -> str:
        """The ID of the MLflow experiment."""
        return self.__experiment_id

    @staticmethod
    def dataset_hash(report: EstimatorReport | CrossValidationReport) -> str:
        """Compute a deterministic hash of report targets for summary compatibility."""
        if hasattr(report, "y_test"):
            return cast(str, hash(report.y_test))
        if hasattr(report, "y"):
            return cast(str, hash(report.y))

        return ""

    @staticmethod
    def _log_model(model: Any) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Any type hint is inferred as AnyType.*",
                category=UserWarning,
            )
            try:
                mlflow.sklearn.log_model(sk_model=model, name="model")
            except TypeError:
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

    def _log_iter(self, iterator: Any, *, log_sub_iters: bool) -> None:
        for obj in iterator:
            if isinstance(obj, tuple):
                if not log_sub_iters:
                    continue
                # TODO: create nested runs
                _subrun_name, sub_iter = obj
                self._log_iter(sub_iter, log_sub_iters=False)
            elif isinstance(obj, Tag):
                mlflow.set_tag(obj.key, obj.value)
            elif isinstance(obj, Params):
                mlflow.log_params(obj.params)
            elif isinstance(obj, Metric):
                mlflow.log_metric(obj.name.replace(".", "_"), obj.value)
            elif callable(getattr(obj, "fit", None)) and callable(
                getattr(obj, "get_params", None)
            ):
                self._log_model(obj)
            elif isinstance(obj, Artifact):
                log_artifact(obj)
            elif isinstance(obj, RawDataArtifact):
                # Raw payload logging is intentionally deferred for now.
                pass
            else:
                raise TypeError(type(obj))

    def put(self, key: str, report: EstimatorReport | CrossValidationReport) -> None:
        """
        Put a key-report pair to the MLflow project.

        Parameters
        ----------
        key : str
            The key to associate with ``report`` in the MLflow project.
        report : skore.EstimatorReport | skore.CrossValidationReport
            The report to associate with ``key`` in the MLflow project.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}').")

        if not isinstance(report, EstimatorReport | CrossValidationReport):
            raise TypeError(
                f"Report must be a `skore.EstimatorReport` or "
                f"`skore.CrossValidationReport` (found '{type(report)}')."
            )

        iterator = (
            iter_estimator(report)
            if isinstance(report, EstimatorReport)
            else iter_cv(report)
        )

        with (
            disable_discrete_autologging(["sklearn"]),
            mlflow.start_run(experiment_id=self.experiment_id, run_name=key),
        ):
            mlflow.set_tags(
                {
                    "skore.learner": report.estimator_name_,
                    "skore.ml_task": report.ml_task,
                    "skore.report_type": report_type(report),
                    "skore.dataset": self.dataset_hash(report),
                    "skore.project_name": self.name,
                }
            )
            self._log_iter(iterator, log_sub_iters=False)

            with TemporaryDirectory() as tmp_dir:
                pickle_path = Path(tmp_dir) / "report.pkl"
                joblib_dump(report, pickle_path)
                mlflow.log_artifact(local_path=str(pickle_path), artifact_path="report")

    def get(self, id: str) -> EstimatorReport | CrossValidationReport:
        """Get a persisted report by its MLflow run id."""
        try:
            pickle_path = mlflow.artifacts.download_artifacts(
                run_id=id,
                artifact_path="report/report.pkl",
                tracking_uri=self.tracking_uri,
            )
        except MlflowException as exc:
            raise KeyError(id) from exc

        return cast(EstimatorReport | CrossValidationReport, joblib_load(pickle_path))

    @staticmethod
    def _metric(metrics: dict[str, float], *keys: str) -> float | None:
        for key in keys:
            if key in metrics:
                return metrics[key]
        return None

    def summarize(self) -> list[Metadata]:
        """Obtain metadata/metrics for all persisted models in insertion order."""
        runs = cast(
            list[Any],
            mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                output_format="list",
                order_by=["attributes.start_time ASC"],
            ),
        )

        return [
            {
                "id": run.info.run_id,
                "key": run.info.run_name,
                "date": format_date(run.info.start_time),
                "learner": run.data.tags.get("skore.learner", ""),
                "ml_task": run.data.tags.get("skore.ml_task", ""),
                "report_type": run.data.tags.get("skore.report_type", ""),
                "dataset": run.data.tags.get("skore.dataset", ""),
                "rmse": self._metric(run.data.metrics, "rmse"),
                "log_loss": self._metric(run.data.metrics, "log_loss"),
                "roc_auc": self._metric(run.data.metrics, "roc_auc"),
                "fit_time": self._metric(run.data.metrics, "fit_time"),
                "predict_time": self._metric(run.data.metrics, "predict_time"),
                "rmse_mean": self._metric(run.data.metrics, "rmse_mean", "rmse"),
                "log_loss_mean": self._metric(
                    run.data.metrics, "log_loss_mean", "log_loss"
                ),
                "roc_auc_mean": self._metric(
                    run.data.metrics, "roc_auc_mean", "roc_auc"
                ),
                "fit_time_mean": self._metric(
                    run.data.metrics, "fit_time_mean", "fit_time"
                ),
                "predict_time_mean": self._metric(
                    run.data.metrics, "predict_time_mean", "predict_time"
                ),
            }
            for run in runs
        ]

    @staticmethod
    def delete(*, name: str) -> None:
        """Not implemented for now."""
        raise NotImplementedError("Delete is not implemented for MLFlow projects")

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"Project(mode='mlflow', name='{self.name}', "
            f"tracking_uri='{self.tracking_uri}')"
        )
