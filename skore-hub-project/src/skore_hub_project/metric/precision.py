"""Class definition of the payload used to send a precision metric to ``hub``."""

from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric, cast_to_float


class Precision(EstimatorReportMetric):  # noqa: D101
    accessor: ClassVar[str] = "metrics.precision"
    name: str = "precision"
    verbose_name: str = "Precision (macro)"
    greater_is_better: bool = True
    position: None = None

    def compute(self) -> None:
        """Compute the value of the metric."""
        try:
            function = self.report.metrics.precision
        except AttributeError:
            self.value = None
        else:
            self.value = cast_to_float(
                function(data_source=self.data_source, average="macro")
            )


class PrecisionTrain(Precision):  # noqa: D101
    data_source: Literal["train"] = "train"


class PrecisionTest(Precision):  # noqa: D101
    data_source: Literal["test"] = "test"


class PrecisionMean(CrossValidationReportMetric):  # noqa: D101
    accessor: ClassVar[str] = "metrics.precision"
    name: str = "precision_mean"
    verbose_name: str = "Precision (macro) - MEAN"
    greater_is_better: bool = True
    position: None = None

    def compute(self) -> None:
        """Compute the value of the metric."""
        try:
            function = self.report.metrics.precision
        except AttributeError:
            self.value = None
        else:
            dataframe = function(
                data_source=self.data_source, aggregate="mean", average="macro"
            )
            self.value = cast_to_float(dataframe.iloc[0, 0])


class PrecisionTrainMean(PrecisionMean):  # noqa: D101
    data_source: Literal["train"] = "train"


class PrecisionTestMean(PrecisionMean):  # noqa: D101
    data_source: Literal["test"] = "test"


class PrecisionStd(CrossValidationReportMetric):  # noqa: D101
    accessor: ClassVar[str] = "metrics.precision"
    name: str = "precision_std"
    verbose_name: str = "Precision (macro) - STD"
    greater_is_better: bool = False
    position: None = None

    def compute(self) -> None:
        """Compute the value of the metric."""
        try:
            function = self.report.metrics.precision
        except AttributeError:
            self.value = None
        else:
            dataframe = function(
                data_source=self.data_source, aggregate="std", average="macro"
            )
            self.value = cast_to_float(dataframe.iloc[0, 0])


class PrecisionTrainStd(PrecisionStd):  # noqa: D101
    data_source: Literal["train"] = "train"


class PrecisionTestStd(PrecisionStd):  # noqa: D101
    data_source: Literal["test"] = "test"
