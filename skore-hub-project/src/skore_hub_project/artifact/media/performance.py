"""Class definition of the payload used to send a performance media to ``hub``."""

from abc import ABC
from collections.abc import Callable
from functools import reduce
from io import BytesIO
from typing import ClassVar, Literal, cast

from matplotlib import pyplot as plt

from skore_hub_project import switch_mpl_backend
from skore_hub_project.artifact.media.media import Media
from skore_hub_project.artifact.serializer import BytesSerializer


class Performance(Media, ABC):  # noqa: D101
    accessor: ClassVar[str]
    serializer: ClassVar[type[BytesSerializer]] = BytesSerializer
    media_type: Literal["image/svg+xml"] = "image/svg+xml"

    def object_to_upload(self) -> bytes | None:
        try:
            function = cast(
                Callable,
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return None

        display = (
            function()
            if self.data_source is None
            else function(data_source=self.data_source)
        )

        with switch_mpl_backend(), BytesIO() as stream:
            display.plot()
            display.figure_.savefig(stream, format="svg", bbox_inches="tight")
            plt.close(display.figure_)

            figure_bytes = stream.getvalue()

        return figure_bytes


class PrecisionRecall(Performance, ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.precision_recall"
    name: Literal["precision_recall"] = "precision_recall"


class PrecisionRecallTrain(PrecisionRecall):  # noqa: D101
    data_source: Literal["train"] = "train"


class PrecisionRecallTest(PrecisionRecall):  # noqa: D101
    data_source: Literal["test"] = "test"


class PredictionError(Performance, ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.prediction_error"
    name: Literal["prediction_error"] = "prediction_error"


class PredictionErrorTrain(PredictionError):  # noqa: D101
    data_source: Literal["train"] = "train"


class PredictionErrorTest(PredictionError):  # noqa: D101
    data_source: Literal["test"] = "test"


class Roc(Performance, ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.roc"
    name: Literal["roc"] = "roc"


class RocTrain(Roc):  # noqa: D101
    data_source: Literal["train"] = "train"


class RocTest(Roc):  # noqa: D101
    data_source: Literal["test"] = "test"
