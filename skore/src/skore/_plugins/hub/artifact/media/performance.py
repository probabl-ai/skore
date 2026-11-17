"""Definition of the payload used to associate a performance media with report."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from functools import reduce
from typing import ClassVar, Literal, cast

from skore import CrossValidationReport, Display
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

        display = function(**self._call_kwargs())
        frame = display.frame(**self._frame_kwargs())

        return dumps(
            frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
        )

    def _call_kwargs(self) -> dict[str, str | None]:
        return {"data_source": self.data_source}

    def _frame_kwargs(self) -> dict[str, str | bool | None]:
        return cast("dict[str, str | bool | None]", self.parameters or {})


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
    parameters: (
        dict[Literal["with_roc_auc"], Literal[True]]
        | dict[Literal["with_roc_auc", "aggregate"], Literal[True, "threshold"]]
    ) = cast(
        "dict[Literal['with_roc_auc'], Literal[True]]",
        {"with_roc_auc": True},
    )


class RocDataFrameTrain(RocDataFrame[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class RocDataFrameTest(RocDataFrame[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class RocDataFrameCrossValidation(RocDataFrame[CrossValidationReport], ABC):  # noqa: D101
    parameters: dict[
        Literal["with_roc_auc", "aggregate"], Literal[True, "threshold"]
    ] = {"with_roc_auc": True, "aggregate": "threshold"}


class RocDataFrameCrossValidationTrain(RocDataFrameCrossValidation):  # noqa: D101
    data_source: Literal["train"] = "train"


class RocDataFrameCrossValidationTest(RocDataFrameCrossValidation):  # noqa: D101
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
