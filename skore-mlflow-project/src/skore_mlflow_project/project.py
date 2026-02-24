"""Class definition of the ``skore`` MLflow project."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from typing import Any, TypedDict, cast

import mlflow
import mlflow.sklearn
from joblib import hash

from .protocol import CrossValidationReport, EstimatorReport


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


def report_type(
    report: EstimatorReport | CrossValidationReport,
) -> str:
    """Human readable type of a report."""
    if hasattr(report, "_report_type"):
        return report._report_type

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
            experiment.experiment_id
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
    def model(report: EstimatorReport | CrossValidationReport) -> Any:
        """Return the fitted model to store in MLflow."""
        if hasattr(report, "estimator_"):
            return report.estimator_

        if hasattr(report, "estimator_reports_"):
            reports = cast(list[EstimatorReport], report.estimator_reports_)
            if reports:
                return reports[0].estimator_

        raise TypeError(
            f"Could not retrieve a fitted estimator from report type '{type(report)}'."
        )

    @staticmethod
    def dataset_hash(report: EstimatorReport | CrossValidationReport) -> str:
        """Compute a deterministic hash of report targets for summary compatibility."""
        if hasattr(report, "y_test"):
            return cast(str, hash(report.y_test))
        if hasattr(report, "y"):
            return cast(str, hash(report.y))

        return ""

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

        with mlflow.start_run(experiment_id=self.experiment_id, run_name=key):
            mlflow.set_tags(
                {
                    "skore.key": key,
                    "skore.learner": report.estimator_name_,
                    "skore.ml_task": report.ml_task,
                    "skore.report_type": report_type(report),
                    "skore.dataset": self.dataset_hash(report),
                    "skore.project_name": self.name,
                }
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Any type hint is inferred as AnyType.*",
                    category=UserWarning,
                )
                try:
                    mlflow.sklearn.log_model(sk_model=self.model(report), name="model")
                except TypeError:
                    mlflow.sklearn.log_model(
                        sk_model=self.model(report), artifact_path="model"
                    )

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
                "key": run.data.tags.get("skore.key", run.info.run_name or ""),
                "date": format_date(run.info.start_time),
                "learner": run.data.tags.get("skore.learner", ""),
                "ml_task": run.data.tags.get("skore.ml_task", ""),
                "report_type": run.data.tags.get("skore.report_type", ""),
                "dataset": run.data.tags.get("skore.dataset", ""),
                "rmse": run.data.metrics.get("rmse"),
                "log_loss": run.data.metrics.get("log_loss"),
                "roc_auc": run.data.metrics.get("roc_auc"),
                "fit_time": run.data.metrics.get("fit_time"),
                "predict_time": run.data.metrics.get("predict_time"),
                "rmse_mean": run.data.metrics.get("rmse_mean"),
                "log_loss_mean": run.data.metrics.get("log_loss_mean"),
                "roc_auc_mean": run.data.metrics.get("roc_auc_mean"),
                "fit_time_mean": run.data.metrics.get("fit_time_mean"),
                "predict_time_mean": run.data.metrics.get("predict_time_mean"),
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
