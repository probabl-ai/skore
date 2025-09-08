"""Class definition of the payload used to send a metric to ``hub``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import suppress
from functools import cached_property, reduce
from math import isfinite
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field

from skore_hub_project.protocol import CrossValidationReport, EstimatorReport


def cast_to_float(value: Any) -> float | None:
    """Cast value to float."""
    with suppress(TypeError):
        if isfinite(value := float(value)):
            return value

    return None


class Metric(ABC, BaseModel):
    """
    Payload used to send a metric to ``hub``.

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
    position: int | None, optional
        Indicator of the "position" of the metric in the parallel coordinates plot,
        default None to disable its display.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    verbose_name: str
    data_source: Literal["train", "test"] | None = None
    greater_is_better: bool | None = None
    position: int | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    @abstractmethod
    def value(self) -> float | None:
        """The value of the metric."""


class EstimatorReportMetric(Metric):
    """
    Payload used to send an estimator report metric.

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
    position: int | None, optional
        Indicator of the "position" of the metric in the parallel coordinates plot,
        default None to disable its display.
    report: EstimatorReport
        The report on which compute the metric.
    accessor : ClassVar[str]
        The "accessor" of the metric i.e., the path to the metric calculation function.
    """

    report: EstimatorReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str]

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:
        """The value of the metric."""
        try:
            function = cast(
                Callable,
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return None

        return cast_to_float(function(data_source=self.data_source))


class CrossValidationReportMetric(Metric):
    """
    Payload used to send a cross-validation report metric, usually MEAN or STD.

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
    position: int | None, optional
        Indicator of the "position" of the metric in the parallel coordinates plot,
        default None to disable its display.
    report: CrossValidationReport
        The report on which compute the metric.
    accessor : ClassVar[str]
        The "accessor" of the metric i.e., the path to the metric calculation function.
    aggregate : ClassVar[Literal["mean", "std"]]
        The aggregation parameter passed to the ``accessor``.
    """

    report: CrossValidationReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str]
    aggregate: ClassVar[Literal["mean", "std"]]

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:
        """The value of the metric."""
        try:
            function = cast(
                Callable,
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return None

        dataframe = function(data_source=self.data_source, aggregate=self.aggregate)

        return cast_to_float(dataframe.iloc[0, 0])
