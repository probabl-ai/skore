"""Class definition of the payload used to send a metric to ``hub``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import suppress
from functools import cached_property, reduce
from math import isfinite
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field, computed_field

from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

if TYPE_CHECKING:
    from pandas import DataFrame

Report = TypeVar("Report", bound=(EstimatorReport | CrossValidationReport))


def cast_to_float(value: Any) -> float | None:
    """Cast value to float."""
    with suppress(TypeError):
        float_value = float(value)

        if isfinite(float_value):
            return float_value

    return None


class Metric(BaseModel, ABC, Generic[Report]):
    """
    Payload used to send a metric to ``hub``.

    Attributes
    ----------
    report : EstimatorReport | CrossValidationReport
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
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    report: Report = Field(repr=False, exclude=True)
    name: str = Field(init=False)
    verbose_name: str = Field(init=False)
    data_source: Literal["train", "test"] | None = Field(init=False)
    greater_is_better: bool | None = Field(init=False)
    position: int | None = Field(init=False)

    @computed_field  # type: ignore[prop-decorator]
    @property
    @abstractmethod
    def value(self) -> float | None:
        """The value of the metric."""


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
    accessor : ClassVar[str]
        The "accessor" of the metric i.e., the path to the metric calculation function.
    """

    accessor: ClassVar[str]

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:
        """The value of the metric."""
        try:
            function = cast(
                Callable[..., float | None],
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return None

        return cast_to_float(function(data_source=self.data_source))


class CrossValidationReportMetric(Metric[CrossValidationReport]):
    """
    Payload used to send a cross-validation report metric, usually MEAN or STD.

    Attributes
    ----------
    report: CrossValidationReport
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
    accessor : ClassVar[str]
        The "accessor" of the metric i.e., the path to the metric calculation function.
    aggregate : ClassVar[Literal["mean", "std"]]
        The aggregation parameter passed to the ``accessor``.
    """

    accessor: ClassVar[str]
    aggregate: ClassVar[Literal["mean", "std"]]

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:
        """The value of the metric."""
        try:
            function = cast(
                "Callable[..., DataFrame]",
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return None

        dataframe = function(data_source=self.data_source, aggregate=self.aggregate)

        return cast_to_float(dataframe.iloc[0, 0])
