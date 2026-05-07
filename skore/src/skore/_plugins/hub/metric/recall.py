"""Class definition of the payload used to send a recall metric to ``hub``."""

from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric, cast_to_float


class Recall(EstimatorReportMetric):  # noqa: D101
    accessor: ClassVar[str] = "metrics.recall"
    name: str = "recall"
    verbose_name: str = "Recall (macro)"
    greater_is_better: bool = True
    position: None = None

    def compute(self) -> None:
        """Compute the value of the metric."""
        try:
            function = self.report.metrics.recall
        except AttributeError:
            self.value = None
        else:
            self.value = cast_to_float(
                function(data_source=self.data_source, average="macro")
            )


class RecallTrain(Recall):  # noqa: D101
    data_source: Literal["train"] = "train"


class RecallTest(Recall):  # noqa: D101
    data_source: Literal["test"] = "test"


class RecallMean(CrossValidationReportMetric):  # noqa: D101
    accessor: ClassVar[str] = "metrics.recall"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: str = "recall_mean"
    verbose_name: str = "Recall (macro) - MEAN"
    greater_is_better: bool = True
    position: None = None

    def compute(self) -> None:
        """Compute the value of the metric."""
        try:
            function = self.report.metrics.recall
        except AttributeError:
            self.value = None
        else:
            dataframe = function(
                data_source=self.data_source, aggregate=self.aggregate, average="macro"
            )
            self.value = cast_to_float(dataframe.iloc[0, 0])


class RecallTrainMean(RecallMean):  # noqa: D101
    data_source: Literal["train"] = "train"


class RecallTestMean(RecallMean):  # noqa: D101
    data_source: Literal["test"] = "test"


class RecallStd(CrossValidationReportMetric):  # noqa: D101
    accessor: ClassVar[str] = "metrics.recall"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: str = "recall_std"
    verbose_name: str = "Recall (macro) - STD"
    greater_is_better: bool = False
    position: None = None

    def compute(self) -> None:
        """Compute the value of the metric."""
        try:
            function = self.report.metrics.recall
        except AttributeError:
            self.value = None
        else:
            dataframe = function(
                data_source=self.data_source, aggregate=self.aggregate, average="macro"
            )
            self.value = cast_to_float(dataframe.iloc[0, 0])


class RecallTrainStd(RecallStd):  # noqa: D101
    data_source: Literal["train"] = "train"


class RecallTestStd(RecallStd):  # noqa: D101
    data_source: Literal["test"] = "test"
