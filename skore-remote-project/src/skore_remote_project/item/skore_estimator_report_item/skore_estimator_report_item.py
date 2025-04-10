"""SkrubTableReportItem."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.utils import estimator_html_repr

from .metadata import Metadata
from ..item import ItemTypeError, lazy_is_instance
from ..matplotlib_figure_item import MatplotlibFigureItem
from ..media_item import MediaItem
from ..pandas_dataframe_item import PandasDataFrameItem
from ..pickle_item import PickleItem

if TYPE_CHECKING:
    from skore import EstimatorReport


REPRESENTATIONS = (
    (
        "performance",
        "precision_recall",
        lambda report: MatplotlibFigureItem.factory(
            report.metrics.precision_recall(data_source="test").plot().figure_
        ),
        {"data_source": "test"},
    ),
    (
        "performance",
        "precision_recall",
        lambda report: MatplotlibFigureItem.factory(
            report.metrics.precision_recall(data_source="train").plot().figure_
        ),
        {"data_source": "train"},
    ),
    (
        "performance",
        "prediction_error",
        lambda report: MatplotlibFigureItem.factory(
            report.metrics.prediction_error(data_source="train").plot().figure_
        ),
        {"data_source": "test"},
    ),
    (
        "performance",
        "prediction_error",
        lambda report: MatplotlibFigureItem.factory(
            report.metrics.prediction_error(data_source="train").plot().figure_
        ),
        {"data_source": "train"},
    ),
    (
        "performance",
        "roc",
        lambda report: MatplotlibFigureItem.factory(
            report.metrics.roc(data_source="train").plot().figure_
        ),
        {"data_source": "test"},
    ),
    (
        "performance",
        "roc",
        lambda report: MatplotlibFigureItem.factory(
            report.metrics.roc(data_source="train").plot().figure_
        ),
        {"data_source": "train"},
    ),
    (
        "feature_importance",
        "permutation",
        lambda report: PandasDataFrameItem.factory(
            report.feature_importance.permutation(data_source="test")
        ),
        {"data_source": "test", "method": "permutation"},
    ),
    (
        "feature_importance",
        "permutation",
        lambda report: PandasDataFrameItem.factory(
            report.feature_importance.permutation(data_source="train")
        ),
        {"data_source": "train", "method": "permutation"},
    ),
    (
        "feature_importance",
        "mean_decrease_impurity",
        lambda report: PandasDataFrameItem.factory(
            report.feature_importance.mean_decrease_impurity()
        ),
        {"method": "mean_decrease_impurity"},
    ),
    (
        "feature_importance",
        "coefficients",
        lambda report: PandasDataFrameItem.factory(
            report.feature_importance.coefficients()
        ),
        {"method": "coefficients"},
    ),
    (
        "model",
        "estimator_html_repr",
        lambda report: MediaItem.factory(
            estimator_html_repr(report.estimator_), media_type="text/html"
        ),
        {},
    ),
)


class SkoreEstimatorReportItem(PickleItem):
    @property
    def __metadata__(self) -> dict[str, float]:
        return dict(Metadata(self.__raw__))

    @property
    def __representation__(self) -> dict[str, dict]:
        report = self.__raw__
        representations = []

        for category, key, function, attributes in REPRESENTATIONS:
            try:
                item = function(report)
            except (AttributeError, ItemTypeError):
                pass
            else:
                representations.append(
                    {
                        "category": category,
                        "key": key,
                        "attributes": attributes,
                        **item.__parameters__,
                        **item.__representation__,
                    }
                )

        return {"related_items": representations}

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
