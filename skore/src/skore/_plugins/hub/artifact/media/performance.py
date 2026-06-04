"""Definition of the payload used to associate a performance media with report."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from functools import reduce
from inspect import signature
from io import BytesIO
from typing import ClassVar, Literal, cast

from matplotlib import pyplot as plt

from skore import Display
from skore._plugins.hub.artifact.media.media import Media, Report
from skore._plugins.hub.json import dumps


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
            fig = display.plot()
            if fig is None:
                # NOTE: backward compatibility for when `figure_` was stored as an
                # attribute in the display object instead of being returned by `plot`.
                fig = display.figure_
            fig.savefig(stream, format="svg", bbox_inches="tight")
            plt.close(fig)

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

        frame = display.frame(**self.get_frame_kwargs(display))

        return dumps(
            frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
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


class PrecisionRecallDataFrame(PerformanceDataFrame[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.precision_recall"
    name: Literal["precision_recall"] = "precision_recall"


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


class RocDataFrameTrain(RocDataFrame[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class RocDataFrameTest(RocDataFrame[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class ConfusionMatrixDataFrame(PerformanceDataFrame[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.confusion_matrix"
    name: Literal["confusion_matrix"] = "confusion_matrix"


class ConfusionMatrixDataFrameTrain(ConfusionMatrixDataFrame[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class ConfusionMatrixDataFrameTest(ConfusionMatrixDataFrame[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"
