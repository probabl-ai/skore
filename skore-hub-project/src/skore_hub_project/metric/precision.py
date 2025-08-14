from __future__ import annotations

from functools import cached_property
from typing import ClassVar, Literal

from pydantic import computed_field

from .metric import CrossValidationReportMetric, EstimatorReportMetric, cast_to_float


class Precision(EstimatorReportMetric):
    accessor: ClassVar[str] = "metrics.precision"
    name: str = "precision"
    verbose_name: str = "Precision (macro)"
    greater_is_better: bool = True

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.precision
        except AttributeError:
            return None

        return cast_to_float(function(data_source=self.data_source, average="macro"))


class PrecisionTrain(Precision):
    data_source: Literal["train"] = "train"


class PrecisionTest(Precision):
    data_source: Literal["test"] = "test"


class PrecisionMean(CrossValidationReportMetric):
    accessor: ClassVar[str] = "metrics.precision"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: str = "precision_mean"
    verbose_name: str = "Precision (macro) - MEAN"
    greater_is_better: bool = True

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.precision
        except AttributeError:
            return None

        dataframe = function(
            data_source=self.data_source, aggregate=self.aggregate, average="macro"
        )

        return cast_to_float(dataframe.iloc[0, 0])


class PrecisionTrainMean(PrecisionMean):
    data_source: Literal["train"] = "train"


class PrecisionTestMean(PrecisionMean):
    data_source: Literal["test"] = "test"


class PrecisionStd(CrossValidationReportMetric):
    accessor: ClassVar[str] = "metrics.precision"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: str = "precision_std"
    verbose_name: str = "Precision (macro) - STD"
    greater_is_better: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.precision
        except AttributeError:
            return None

        dataframe = function(
            data_source=self.data_source, aggregate=self.aggregate, average="macro"
        )

        return cast_to_float(dataframe.iloc[0, 0])


class PrecisionTrainStd(PrecisionStd):
    data_source: Literal["train"] = "train"


class PrecisionTestStd(PrecisionStd):
    data_source: Literal["test"] = "test"
