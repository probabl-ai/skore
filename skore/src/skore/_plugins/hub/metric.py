"""Class definition of the payload used to send a metric to ``hub``."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from skore import CrossValidationReport, EstimatorReport

Report = TypeVar("Report", bound=(EstimatorReport | CrossValidationReport))


class Metric(BaseModel, Generic[Report]):
    """
    Payload used to send a metric.

    Attributes
    ----------
    name : str
        Name of the metric.
    verbose_name : str
        Verbose name of the metric.
    data_source : Literal["train", "test"] | None, optional
        Data source of the metric when it can be declined in several ways, default None.
    greater_is_better: bool | None, optional
        Indicator of "greater value is better", default None.
    value : float
        Value of the metric.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    verbose_name: str
    data_source: Literal["train", "test"] | None
    greater_is_better: bool | None
    value: float
    # See https://github.com/probabl-ai/skore/issues/3025
    position: None = Field(default=None)
