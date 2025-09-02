from contextlib import suppress
from dataclasses import InitVar, dataclass, field, fields
from datetime import datetime, timezone
from math import isfinite
from typing import Any

import joblib
from skore import CrossValidationReport, EstimatorReport


def cast_to_float(value: Any) -> float | None:
    """Cast value to float."""
    with suppress(TypeError):
        if isfinite(value := float(value)):
            return value

    return None


def report_type(report):
    if isinstance(report, CrossValidationReport):
        return "cross-validation"
    if isinstance(report, EstimatorReport):
        return "estimator"

    raise TypeError


@dataclass
class EstimatorReportMetadata:
    report: InitVar[EstimatorReport]

    artifact_id: str
    project_name: str
    run_id: str
    key: str
    date: str = field(init=False)
    learner: str = field(init=False)
    dataset: str = field(init=False)
    ml_task: str = field(init=False)
    report_type: str = field(init=False)
    rmse: float | None = field(init=False)
    log_loss: float | None = field(init=False)
    roc_auc: float | None = field(init=False)
    fit_time: float | None = field(init=False)
    predict_time: float | None = field(init=False)

    @staticmethod
    def metric(report: EstimatorReport, name: str) -> float | None:
        if not hasattr(report.metrics, name):
            return None

        return cast_to_float(getattr(report.metrics, name)(data_source="test"))

    def __post_init__(self, report: EstimatorReport):
        self.date = datetime.now(timezone.utc).isoformat()
        self.learner = report.estimator_name_
        self.dataset = joblib.hash(report.y_test)
        self.ml_task = report.ml_task
        self.report_type = report_type(report)
        self.rmse = self.metric(report, "rmse")
        self.log_loss = self.metric(report, "log_loss")
        self.roc_auc = self.metric(report, "roc_auc")
        # timings must be calculated last
        self.fit_time = report.metrics.timings().get("fit_time")
        self.predict_time = report.metrics.timings().get("predict_time_test")

    def __iter__(self):
        for field in fields(self):
            yield (field.name, getattr(self, field.name))


@dataclass
class CrossValidationReportMetadata:
    report: InitVar[CrossValidationReport]

    artifact_id: str
    project_name: str
    run_id: str
    key: str
    date: str = field(init=False)
    learner: str = field(init=False)
    dataset: str = field(init=False)
    ml_task: str = field(init=False)
    report_type: str = field(init=False)
    rmse_mean: float | None = field(init=False)
    log_loss_mean: float | None = field(init=False)
    roc_auc_mean: float | None = field(init=False)
    fit_time_mean: float | None = field(init=False)
    predict_time_mean: float | None = field(init=False)

    @staticmethod
    def metric(report: CrossValidationReport, name: str) -> float | None:
        if not hasattr(report.metrics, name):
            return None

        dataframe = getattr(report.metrics, name)(
            data_source="test",
            aggregate="mean",
        )

        return cast_to_float(dataframe.iloc[0, 0])

    @staticmethod
    def timing(report: CrossValidationReport, label: str) -> float | None:
        dataframe = report.metrics.timings(aggregate="mean")

        try:
            series = dataframe.loc[label]
        except KeyError:
            return None

        return cast_to_float(series.iloc[0])

    def __post_init__(self, report: CrossValidationReport):
        self.date = datetime.now(timezone.utc).isoformat()
        self.learner = report.estimator_name_
        self.dataset = joblib.hash(report.y)
        self.ml_task = report.ml_task
        self.report_type = report_type(report)
        self.rmse_mean = self.metric(report, "rmse")
        self.log_loss_mean = self.metric(report, "log_loss")
        self.roc_auc_mean = self.metric(report, "roc_auc")
        # timings must be calculated last
        self.fit_time_mean = self.timing(report, "Fit time (s)")
        self.predict_time_mean = self.timing(report, "Predict time test (s)")

    def __iter__(self):
        for field in fields(self):
            yield (field.name, getattr(self, field.name))
