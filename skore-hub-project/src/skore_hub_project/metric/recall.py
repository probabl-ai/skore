"""Class definition of the payload used to send a recall metric to ``hub``."""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar, Literal

from pydantic import computed_field

from .metric import CrossValidationReportMetric, EstimatorReportMetric, cast_to_float


class Recall(EstimatorReportMetric):  # noqa: D101
    accessor: ClassVar[str] = "metrics.recall"
    name: str = "recall"
    verbose_name: str = "Recall (macro)"
    greater_is_better: bool = True
    position: None = None

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:  # noqa: D102
        try:
            function = self.report.metrics.recall
        except AttributeError:
            return None

        return cast_to_float(function(data_source=self.data_source, average="macro"))


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

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:  # noqa: D102
        try:
            function = self.report.metrics.recall
        except AttributeError:
            return None

        dataframe = function(
            data_source=self.data_source, aggregate=self.aggregate, average="macro"
        )

        return cast_to_float(dataframe.iloc[0, 0])


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

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:  # noqa: D102
        try:
            function = self.report.metrics.recall
        except AttributeError:
            return None

        dataframe = function(
            data_source=self.data_source, aggregate=self.aggregate, average="macro"
        )

        return cast_to_float(dataframe.iloc[0, 0])


class RecallTrainStd(RecallStd):  # noqa: D101
    data_source: Literal["train"] = "train"


class RecallTestStd(RecallStd):  # noqa: D101
    data_source: Literal["test"] = "test"
