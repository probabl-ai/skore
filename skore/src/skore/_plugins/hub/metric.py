"""Class definition of the payload used to send a metric to ``hub``."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from skore import CrossValidationReport, EstimatorReport

Report = TypeVar("Report", bound=(EstimatorReport | CrossValidationReport))


class Metric(BaseModel, Generic[Report]):
    """Payload used to send a metric to ``hub``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    report: Report = Field(repr=False, exclude=True)
    name: str
    verbose_name: str
    data_source: Literal["train", "test"] | None
    greater_is_better: bool | None
    # TODO: Remove the attribute if Hub no longer requires it
    position: None = Field(default=None)


class EstimatorReportMetric(Metric[EstimatorReport]):
    """
    Payload used to send an estimator report metric.

    Attributes
    ----------
    report: EstimatorReport
        The report on which compute the metric.
    name : str
        Name of the metric.
    verbose_name : str
        Verbose name of the metric.
    data_source : Literal["train", "test"] | None, optional
        Data source of the metric when it can be declined in several ways, default None.
    greater_is_better: bool | None, optional
        Indicator of "greater value is better", default None.
    position: int | None, optional
        Indicator of the "position" of the metric in the parallel coordinates plot,
        default None to disable its display.
    value : float or None
        Value of the metric
    """

    value: float | None


class CrossValidationReportMetric(Metric[CrossValidationReport]):
    """
    Payload used to send a cross-validation report metric, usually MEAN or STD.

    Notes
    -----
    The aggregated value (mean or std) is sent under ``value``; ``name`` carries
    the aggregation suffix (e.g. ``"accuracy_mean"`` / ``"accuracy_std"``). This
    matches the wire format the hub currently expects. A future version will
    likely send ``mean`` and ``std`` side by side under a single metric entry.

    Attributes
    ----------
    report: CrossValidationReport
        The report on which compute the metric.
    name : str
        Name of the metric, with an aggregation suffix.
    verbose_name : str
        Verbose name of the metric, with an aggregation suffix.
    data_source : Literal["train", "test"] | None, optional
        Data source of the metric when it can be declined in several ways, default None.
    greater_is_better : bool | None, optional
        Indicator of "greater value is better", default None.
    value : float
        Aggregated metric value (mean or std).
    """

    value: float
