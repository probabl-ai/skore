"""Definition of the payload used to associate a performance media with report."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from functools import reduce
from typing import ClassVar, Literal, cast

from skore import Display
from skore._plugins.hub.artifact.media.media import Media, Report
from skore._plugins.hub.json import dumps


class PerformanceDataFrame(Media[Report], ABC):  # noqa: D101
    accessor: ClassVar[str]
    content_type: Literal["application/vnd.dataframe"] = "application/vnd.dataframe"

    def content_to_upload(self) -> bytes | None:  # noqa: D102
        try:
            function = cast(
                "Callable[..., Display]",
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return None

        display = function(data_source=self.data_source)
        frame = display.frame(**(self.parameters or {}))  # type: ignore[arg-type]

        return dumps(
            frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
        )


class PrecisionRecallDataFrame(PerformanceDataFrame[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.precision_recall"
    name: Literal["precision_recall"] = "precision_recall"
    parameters: dict[Literal["with_average_precision"], Literal[True]] = {
        "with_average_precision": True
    }


class PrecisionRecallDataFrameTrain(PrecisionRecallDataFrame[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class PrecisionRecallDataFrameTest(PrecisionRecallDataFrame[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class PredictionErrorDataFrame(PerformanceDataFrame[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.prediction_error"
    name: Literal["prediction_error"] = "prediction_error"


class PredictionErrorDataFrameTrain(PredictionErrorDataFrame[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class PredictionErrorDataFrameTest(PredictionErrorDataFrame[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class RocDataFrame(PerformanceDataFrame[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.roc"
    name: Literal["roc"] = "roc"
    parameters: dict[Literal["with_roc_auc"], Literal[True]] = {"with_roc_auc": True}


class RocDataFrameTrain(RocDataFrame[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class RocDataFrameTest(RocDataFrame[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class ConfusionMatrixDataFrame(PerformanceDataFrame[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.confusion_matrix"
    name: Literal["confusion_matrix"] = "confusion_matrix"


class ConfusionMatrixDataFrameTrainAll(ConfusionMatrixDataFrame[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"
    parameters: dict[Literal["threshold_value"], Literal["all"]] = {
        "threshold_value": "all"
    }


class ConfusionMatrixDataFrameTrainNone(ConfusionMatrixDataFrame[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"
    parameters: dict[Literal["threshold_value"], None] = {"threshold_value": None}


class ConfusionMatrixDataFrameTestAll(ConfusionMatrixDataFrame[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"
    parameters: dict[Literal["threshold_value"], Literal["all"]] = {
        "threshold_value": "all"
    }


class ConfusionMatrixDataFrameTestNone(ConfusionMatrixDataFrame[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"
    parameters: dict[Literal["threshold_value"], None] = {"threshold_value": None}
