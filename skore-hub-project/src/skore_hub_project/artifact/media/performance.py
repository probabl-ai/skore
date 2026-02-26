"""Definition of the payload used to associate a performance media with report."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from functools import reduce
from io import BytesIO
from typing import ClassVar, Literal, cast

from matplotlib import pyplot as plt

from skore_hub_project.artifact.media.media import Media, Report
from skore_hub_project.json import dumps
from skore_hub_project.protocol import Display


class PerformanceSVG(Media[Report], ABC):  # noqa: D101
    accessor: ClassVar[str]
    content_type: Literal["image/svg+xml"] = "image/svg+xml"

    def content_to_upload(self) -> bytes | None:  # noqa: D102
        try:
            function = cast(
                "Callable[..., Display]",
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return None

        display = (
            function()
            if self.data_source is None
            else function(data_source=self.data_source)
        )

        with BytesIO() as stream:
            display.plot()
            display.figure_.savefig(stream, format="svg", bbox_inches="tight")  # type: ignore[attr-defined]
            plt.close(display.figure_)  # type: ignore[attr-defined]

            figure_bytes = stream.getvalue()

        return figure_bytes


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

        display = (
            function()
            if self.data_source is None
            else function(data_source=self.data_source)
        )

        frame = display.frame()

        return dumps(
            frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
        )


class PrecisionRecallSVG(PerformanceSVG[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.precision_recall"
    name: Literal["precision_recall"] = "precision_recall"


class PrecisionRecallDataFrame(PerformanceDataFrame[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.precision_recall"
    name: Literal["precision_recall"] = "precision_recall"


class PrecisionRecallSVGTrain(PrecisionRecallSVG[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class PrecisionRecallDataFrameTrain(PrecisionRecallDataFrame[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class PrecisionRecallSVGTest(PrecisionRecallSVG[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class PrecisionRecallDataFrameTest(PrecisionRecallDataFrame[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class PredictionErrorSVG(PerformanceSVG[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.prediction_error"
    name: Literal["prediction_error"] = "prediction_error"


class PredictionErrorDataFrame(PerformanceDataFrame[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.prediction_error"
    name: Literal["prediction_error"] = "prediction_error"


class PredictionErrorSVGTrain(PredictionErrorSVG[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class PredictionErrorDataFrameTrain(PredictionErrorDataFrame[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class PredictionErrorSVGTest(PredictionErrorSVG[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class PredictionErrorDataFrameTest(PredictionErrorDataFrame[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class RocSVG(PerformanceSVG[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.roc"
    name: Literal["roc"] = "roc"


class RocDataFrame(PerformanceDataFrame[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.roc"
    name: Literal["roc"] = "roc"


class RocSVGTrain(RocSVG[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class RocDataFrameTrain(RocDataFrame[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class RocSVGTest(RocSVG[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class RocDataFrameTest(RocDataFrame[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class ConfusionMatrixSVG(PerformanceSVG[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.confusion_matrix"
    name: Literal["confusion_matrix"] = "confusion_matrix"


class ConfusionMatrixDataFrame(PerformanceDataFrame[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.confusion_matrix"
    name: Literal["confusion_matrix"] = "confusion_matrix"


class ConfusionMatrixSVGTrain(ConfusionMatrixSVG[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class ConfusionMatrixDataFrameTrain(ConfusionMatrixDataFrame[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class ConfusionMatrixSVGTest(ConfusionMatrixSVG[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class ConfusionMatrixDataFrameTest(ConfusionMatrixDataFrame[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"
