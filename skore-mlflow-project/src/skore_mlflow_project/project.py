"""Class definition of the ``skore`` MLflow project."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict, cast

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.entities import Run as MLFlowRun
from mlflow.exceptions import MlflowException
from mlflow.utils.autologging_utils import disable_discrete_autologging
from sklearn.base import BaseEstimator

from .protocol import CrossValidationReport, EstimatorReport
from .reports import (
    Artifact,
    Metric,
    Model,
    NestedLogItem,
    Params,
    Tag,
    iter_cv,
    iter_estimator,
)


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

    return datetime.fromtimestamp(start_time / 1_000, tz=timezone.utc).isoformat()


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
                    "skore_version": version("skore"),
                    "report_type": report_type(report),
                    "ml_task": report.ml_task,
                    "learner": report.estimator_name_,
                }
            )
            _log_iter(iterator)

            with TemporaryDirectory() as tmp_dir:
                pickle_path = Path(tmp_dir) / "report.pkl"
                with _cache_cleared_for_pickle(report):
                    joblib.dump(report, pickle_path)
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

        return cast(EstimatorReport | CrossValidationReport, joblib.load(pickle_path))

    def summarize(self) -> list[Metadata]:
        """Obtain metadata/metrics for all persisted models in insertion order."""
        runs = cast(
            list[MLFlowRun],
            mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                output_format="list",
                order_by=["attributes.start_time ASC"],
                filter_string='tags.skore_version != ""',
            ),
        )

        return [self._run_to_metadata(run) for run in runs]

    @staticmethod
    def _run_to_metadata(run: MLFlowRun) -> Metadata:
        tags = run.data.tags
        metrics = run.data.metrics
        report_type = tags["report_type"]

        metadata = {
            "id": run.info.run_id,
            "key": run.info.run_name,
            "date": format_date(run.info.start_time),
            "report_type": report_type,
            "learner": tags["learner"],
            "ml_task": tags["ml_task"],
            "dataset": "",  # TODO
            "rmse": None,
            "log_loss": None,
            "roc_auc": None,
            "fit_time": None,
            "predict_time": None,
            "rmse_mean": None,
            "log_loss_mean": None,
            "roc_auc_mean": None,
            "fit_time_mean": None,
            "predict_time_mean": None,
        }

        if report_type == "estimator":
            metrics = {
                "rmse": run.data.metrics.get("rmse"),
                "log_loss": run.data.metrics.get("log_loss"),
                "roc_auc": run.data.metrics.get("roc_auc"),
                "fit_time": metrics["fit_time"],
                "predict_time": metrics["predict_time"],
            }
        elif report_type == "cross-validation":
            metrics = {
                "rmse_mean": run.data.metrics.get("rmse"),
                "log_loss_mean": run.data.metrics.get("log_loss"),
                "roc_auc_mean": run.data.metrics.get("roc_auc"),
                "fit_time_mean": metrics["fit_time"],
                "predict_time_mean": metrics["predict_time"],
            }
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
        return cast(Metadata, {**metadata, **metrics})

    @staticmethod
    def delete(*, name: str, tracking_uri: str | None = None) -> None:
        """Not implemented for now."""
        raise NotImplementedError("Delete is not implemented for MLFlow projects")

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"Project(mode='mlflow', name='{self.name}', "
            f"tracking_uri='{self.tracking_uri}')"
        )


## Helpers for logging in MLFlow:


def _log_iter(iterator: Iterable[NestedLogItem]) -> None:
    for obj in iterator:
        if isinstance(obj, tuple):
            subrun_name, sub_iter = obj
            with mlflow.start_run(nested=True, run_name=subrun_name):
                _log_iter(sub_iter)
        elif isinstance(obj, Tag):
            mlflow.set_tag(obj.key, obj.value)
        elif isinstance(obj, Params):
            mlflow.log_params(obj.params)
        elif isinstance(obj, Metric):
            mlflow.log_metric(obj.name.replace(".", "_"), obj.value)
        elif isinstance(obj, Model):
            _log_model(obj.model)
        elif isinstance(obj, Artifact):
            _log_artifact(obj)
        else:
            raise TypeError(type(obj))


def _log_model(model: BaseEstimator) -> None:
    try:
        with (
            # MLflow 3.10 emits this warning (and not 3.9), looks like a bug:
            _filterwarnings(UserWarning, ".*Any type hint is inferred as AnyType.*"),
            # MLflow 3.9 emits this warning but not 3.10...
            _filterwarnings(FutureWarning, ".*pickle.*format.*arbitrary code.*"),
            # MLflow 3.4 on SQLAlchemy 2 emits this warning internally.
            _filterwarnings(Warning, r".*Query.get\(\) method is considered legacy.*"),
            # MLflow <= 3.2 emits this warning internally.
            _filterwarnings(DeprecationWarning, r".*utcnow\(\) is deprecated.*"),
            # MLflow 3.0 can trigger this from pydantic internals.
            _filterwarnings(Warning, ".*@model_validator.*deprecated.*"),
        ):
            mlflow.sklearn.log_model(model, name="model")
    except TypeError as exc:
        # MLflow < 3 does not support the `name` parameter.
        if "unexpected keyword argument 'name'" not in str(exc):
            raise
        with _filterwarnings(DeprecationWarning, r".*utcnow\(\) is deprecated.*"):
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                # Avoid importing MLflow's pyfunc flavor path on MLflow 2, which emits
                # deprecation warnings with modern pydantic versions:
                pyfunc_predict_fn="__skore_disabled_pyfunc__",
            )


@contextmanager
def _filterwarnings(category: type[Warning], msg: str) -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=msg,
            category=category,
        )
        yield


@contextmanager
def _cache_cleared_for_pickle(
    report: EstimatorReport | CrossValidationReport,
) -> Iterator[None]:
    # Keep track of nested estimator report caches to restore local object state.
    reports = [report] + getattr(report, "estimator_reports_", [])
    caches = [report_to_clear._cache for report_to_clear in reports]
    report.clear_cache()

    try:
        yield
    finally:
        for report_to_restore, cache in zip(reports, caches, strict=True):
            report_to_restore._cache = cache


def _log_artifact(artifact: Artifact) -> None:
    """Log a report artifact."""

    def filename(ext: str) -> str:
        return f"report/{artifact.name}.{ext}"

    payload = artifact.payload
    if callable(getattr(payload, "to_csv", None)):
        csv_text = _flatten_df_index(payload).to_csv(index=False)
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


def _flatten_df_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a dataframe-like object before CSV logging."""
    df = df.copy(deep=False)
    columns = df.columns
    if columns is not None and columns.nlevels > 1:
        df.columns = columns.droplevel(0)

    index = df.index
    if isinstance(index, pd.RangeIndex) and len(index.names) == 1:
        return df

    return df.reset_index()


HTML_UTF8_TEMPLATE = """
<head>
<meta charset="UTF-8">
</head>
<body>
{html}
</body>
"""


def _wrap_html(html: str) -> str:
    return HTML_UTF8_TEMPLATE.format(html=html)
