"""SkrubTableReportItem."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import joblib

from .item import ItemTypeError, lazy_is_instance
from .pickle_item import PickleItem

if TYPE_CHECKING:
    from skore import EstimatorReport


METRICS = (
    "accuracy",
    "brier_score",
    "log_loss",
    "precision",
    "r2",
    "recall",
    "rmse",
    "roc_auc",
)


PLOTS = (
    "precision_recall",
    "prediction_error",
    "roc",
)


class SkoreEstimatorReportItem(PickleItem):
    @property
    def __metadata__(self) -> dict[str, float]:
        report = self.__raw__
        estimator = report.estimator_  # pipeline not implemented yet

        return {
            "pipeline": {estimator.__class__.__name__: estimator.get_params()},
            "dataset_fingerprint": joblib.hash(
                (
                    report.X_train,
                    report.y_train,
                    report.X_test,
                    report.y_test,
                )
            ),
            "ml_task": report.ml_task,  # modify hub's allowed values
            "metrics": [
                {
                    "name": metric_name,
                    "value": getattr(report.metrics, metric_name)(data_source=source),
                    "data_source": source,
                    "greater_is_better": None,  # how to get efficiently?
                }
                for metric_name, source in itertools.product(METRICS, ("test", "train"))
                if hasattr(report.metrics, metric_name)
            ],
        }

    @property
    def __representation__(self) -> dict[str, dict]:
        from . import object_to_item

        report = self.__raw__
        representation = {}

        for plot_name, source in itertools.product(PLOTS, ("test", "train")):
            if hasattr(report.metrics, plot_name):
                plot = getattr(report.metrics, plot_name)(data_source=source)
                item = object_to_item(plot)

                representation[plot_name] = {
                    "data_source": source,
                    **item.__parameters__,
                    **item.__representation__,
                }

        return {"related_items": representation}

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
