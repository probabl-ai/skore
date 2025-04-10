from __future__ import annotations

from typing import TYPE_CHECKING

from ..item import ItemTypeError, lazy_is_instance
from ..pickle_item import PickleItem
from .metadata import Metadata
from .representation import Representation

if TYPE_CHECKING:
    from skore import EstimatorReport


class SkoreEstimatorReportItem(PickleItem):
    @property
    def __metadata__(self) -> dict[str, float]:
        return dict(Metadata(self.__raw__))

    @property
    def __representation__(self) -> dict[str, dict]:
        return {"related_items": dict(Representation(self.__raw__))}

    @classmethod
    def factory(cls, report: EstimatorReport, /) -> SkoreEstimatorReportItem:
        """
        Create a new SkoreEstimatorReportItem from a skore ``EstimatorReport``.

        Parameters
        ----------
        report : EstimatorReport
            The report to store.

        Returns
        -------
        SkoreEstimatorReportItem
            A new SkoreEstimatorReportItem instance.
        """
        if lazy_is_instance(report, "skore.sklearn._estimator.report.EstimatorReport"):
            return super().factory(report)

        raise ItemTypeError(f"Type '{report.__class__}' is not supported.")
