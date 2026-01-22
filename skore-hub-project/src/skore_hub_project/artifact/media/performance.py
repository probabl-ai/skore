"""Definition of the payload used to associate a performance media with report."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from functools import reduce
from io import BytesIO
from multiprocessing import Event, get_context
from typing import ClassVar, Literal, cast

from matplotlib import pyplot as plt

from skore_hub_project.artifact.media.media import Media, Report
from skore_hub_project.protocol import Display

# Matplotlib is not thread-safe: https://matplotlib.org/stable/users/faq.html#work-with-threads.
#
# Even with the non-interactive backend ``Agg``, when we try to create the same
# plot from different threads, we encounter problems with auto-scaling and they
# have inconsistent axis scales.


def plot(queue, /):
    while task := queue.get():
        # raise Exception()
        # how to interrupt main on exception?

        filepath, display = task

        with BytesIO() as stream:
            display.plot()
            display.figure_.savefig(stream, format="svg", bbox_inches="tight")
            plt.close(display.figure_)

            filepath.write_bytes(stream.getvalue())

        queue.task_done()


ctx = get_context("fork")
PLOTTER_QUEUE = ctx.JoinableQueue()
PLOTTER = ctx.Process(target=plot, args=(PLOTTER_QUEUE,), daemon=True)
PLOTTER.start()


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

        PLOTTER_QUEUE.put(
            (
                self.filepath,
                (
                    function()
                    if self.data_source is None
                    else function(data_source=self.data_source)
                ),
            )
        )

        # assert self.filepath.stat().st_size

        while not self.filepath.stat().st_size:
            ...

        # timeout raise exception


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
