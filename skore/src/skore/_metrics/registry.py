from __future__ import annotations

from collections import OrderedDict, UserDict
from typing import TYPE_CHECKING, Literal

from skore._metrics.metrics import BUILTIN_METRICS, Metric, Score

if TYPE_CHECKING:
    from skore import EstimatorReport


class MetricRegistry(UserDict[str, Metric]):
    """Registry of metric instances for a report.

    Parameters
    ----------
    report : EstimatorReport
        The parent report.
    """

    data: OrderedDict[str, Metric]

    def __init__(self, report: EstimatorReport) -> None:
        """Construct a MetricRegistry.

        The report is analyzed to filter metrics depending on the report's
        characteristics (e.g. the ML task and the estimator's prediction methods).
        """
        super().__init__()

        # Needs to be called ``data`` since we inherit from :class:`UserDict`
        self.data = OrderedDict(
            (metric.name, metric)
            for metric in BUILTIN_METRICS
            if metric.available(report)
        )

        if Score.available(report):
            self.data["score"] = Score()
            self.data.move_to_end("score", last=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self.data.keys())})"

    def add(
        self,
        metric: Metric,
        *,
        position: Literal["first", "last"] = "first",
    ) -> None:
        """Add a custom metric to the registry.

        Parameters
        ----------
        metric : Metric
            The metric instance to add.

        position : {"first", "last"}, default="first"
            Where to place the metric in iteration order (e.g. default
            :meth:`~skore.EstimatorReport.metrics.summarize` row order).
            ``"first"`` inserts at the front; ``"last"`` at the end.
        """
        if position not in ("first", "last"):
            raise ValueError(f"position must be 'first' or 'last', got {position!r}.")

        if metric.name == "score":
            raise ValueError(f"Cannot add {metric.name!r}: it is a reserved name.")

        if metric.name in self.data:
            raise ValueError(
                f"Cannot add {metric.name!r}: it already exists. "
                "Remove it first using the `remove` method."
            )

        self.data[metric.name] = metric

        if position == "first":
            self.data.move_to_end(metric.name, last=False)

    def remove(self, *, report: EstimatorReport, name: str) -> None:
        """Remove a metric from the registry.

        Built-in metrics may be removed; they stay absent for the lifetime of this
        registry (the same instance is kept for the parent report).

        Parameters
        ----------
        name : str
            The technical name of the metric to remove.

        Raises
        ------
        KeyError
            If `name` is not registered.
        """
        del self.data[name]

        keys_to_delete = [k for k in report._cache if k[2] == name]
        for k in keys_to_delete:
            del report._cache[k]
