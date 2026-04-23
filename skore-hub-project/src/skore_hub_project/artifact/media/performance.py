"""Definition of the payload used to associate a performance media with report."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from functools import reduce
from inspect import signature
from io import BytesIO
from typing import ClassVar, Literal, cast

from matplotlib import pyplot as plt

from skore_hub_project.artifact.media.media import Media, Report
from skore_hub_project.json import dumps
from skore_hub_project.protocol import Display


class PerformanceSVG(Media[Report], ABC):  # noqa: D101
    accessor: ClassVar[str]
    content_type: Literal["image/svg+xml"] = "image/svg+xml"

    def compute(self) -> None:  # noqa: D102
        if self.computed:
            return

        self.computed = True

        try:
            function = cast(
                "Callable[..., Display]",
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return

        display = (
            function()
            if self.data_source is None
            else function(data_source=self.data_source)
        )

        with BytesIO() as stream:
            fig = display.plot()

            if fig is None:
                # NOTE: backward compatibility for when `figure_` was stored as an
                # attribute in the display object instead of being returned by `plot`.
                fig = display.figure_

            fig.savefig(stream, format="svg", bbox_inches="tight")
            plt.close(fig)

            self.filepath.write_bytes(stream.getvalue())


class PerformanceDataFrame(Media[Report], ABC):  # noqa: D101
    accessor: ClassVar[str]
    content_type: Literal["application/vnd.dataframe"] = "application/vnd.dataframe"

    def compute(self) -> None:  # noqa: D102
        if self.computed:
            return

        self.computed = True

        try:
            function = cast(
                "Callable[..., Display]",
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return

        display = (
            function()
            if self.data_source is None
            else function(data_source=self.data_source)
        )

        frame = display.frame(**self.get_frame_kwargs(display))

        self.filepath.write_bytes(
            dumps(
                frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
            )
        )

    @staticmethod
    def get_frame_kwargs(display: Display) -> dict[str, str | bool]:
        """Get the kwargs to pass to the frame method."""
        params = signature(display.frame).parameters
        kwargs: dict[str, str | bool] = {}
        if "threshold_value" in params:
            kwargs["threshold_value"] = "all"
        if "with_average_precision" in params:
            kwargs["with_average_precision"] = True
        if "with_roc_auc" in params:
            kwargs["with_roc_auc"] = True

        return kwargs


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
