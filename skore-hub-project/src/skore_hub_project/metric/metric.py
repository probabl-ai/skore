from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from functools import cached_property, reduce
from math import isfinite
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, computed_field
from skore import CrossValidationReport, EstimatorReport


def cast_to_float(value: Any) -> float | None:
    """Cast value to float."""
    with suppress(TypeError):
        if isfinite(value := float(value)):
            return value

    return None


class Metric(ABC, BaseModel):
    name: str
    verbose_name: str
    data_source: Literal["train", "test"] | None = None
    greater_is_better: bool | None = None
    position: int | None = None

    class Config:
        frozen = True

    @computed_field
    @property
    @abstractmethod
    def value(self) -> float | None: ...


class EstimatorReportMetric(Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str]

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = reduce(getattr, self.accessor.split("."), self.report)
        except AttributeError:
            return None

        return cast_to_float(function(data_source=self.data_source))


class CrossValidationReportMetric(Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str]
    aggregate: ClassVar[Literal["mean", "std"]]

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = reduce(getattr, self.accessor.split("."), self.report)
        except AttributeError:
            return None

        dataframe = function(data_source=self.data_source, aggregate=self.aggregate)

        return cast_to_float(dataframe.iloc[0, 0])
