from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from functools import cached_property, reduce
from math import isfinite
from typing import Any, ClassVar, Literal

from pydantic import Field, computed_field

from skore_hub_project import Payload

CrossValidationReport = Any
EstimatorReport = Any


def cast_to_float(value: Any) -> float | None:
    """Cast value to float."""
    with suppress(TypeError):
        if isfinite(value := float(value)):
            return value

    return None


class Metric(ABC, Payload):
    name: str
    verbose_name: str
    data_source: Literal["train", "test"] | None = None
    greater_is_better: bool | None = None
    position: int | None = None

    @computed_field
    @property
    @abstractmethod
    def value(self) -> float | None: ...

    def model_dump(self, *args, **kwargs):
        if self.value is None:
            return None

        return super().model_dump(*args, **kwargs)


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
