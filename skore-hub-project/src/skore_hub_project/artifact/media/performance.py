"""Definition of the payload used to associate a performance media with report."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from functools import reduce
from io import BytesIO
from typing import ClassVar, Literal, cast

from matplotlib import pyplot as plt

from skore_hub_project import switch_mpl_backend
from skore_hub_project.artifact.media.media import Media, Report
from skore_hub_project.protocol import Display


class Performance(Media[Report], ABC):  # noqa: D101
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

        with switch_mpl_backend(), BytesIO() as stream:
            display.plot()
            display.figure_.savefig(stream, format="svg", bbox_inches="tight")  # type: ignore[attr-defined]
            plt.close(display.figure_)  # type: ignore[attr-defined]

            self.filepath.write_bytes(stream.getvalue())


class PrecisionRecall(Performance[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.precision_recall"
    name: Literal["precision_recall"] = "precision_recall"


class PrecisionRecallTrain(PrecisionRecall[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class PrecisionRecallTest(PrecisionRecall[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class PredictionError(Performance[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.prediction_error"
    name: Literal["prediction_error"] = "prediction_error"


class PredictionErrorTrain(PredictionError[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class PredictionErrorTest(PredictionError[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class Roc(Performance[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "metrics.roc"
    name: Literal["roc"] = "roc"


class RocTrain(Roc[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class RocTest(Roc[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"
