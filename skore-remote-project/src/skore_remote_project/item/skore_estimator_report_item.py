"""SkrubTableReportItem."""

from __future__ import annotations

import itertools
from numbers import Number
from typing import TYPE_CHECKING, Any

from joblib import hash as joblib_hash
from matplotlib.pyplot import subplots

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


def is_primitive(obj: Any) -> bool:
    """Check if the object is a primitive."""
    if isinstance(obj, (type(None), bool, float, int, str)):
        return True
    if isinstance(obj, (list, tuple)):
        return all(is_primitive(item) for item in obj)
    if isinstance(obj, dict):
        return all(
            isinstance(k, (bool, float, int, str)) and is_primitive(v)
            for k, v in obj.items()
        )
    return False


class SkoreEstimatorReportItem(PickleItem):
    @property
    def __metadata__(self) -> dict[str, float]:
        report = self.__raw__
        estimator = report.estimator_
        metrics = []

        for metric_name, source in itertools.product(METRICS, ("test", "train")):
            if not hasattr(report.metrics, metric_name):
                continue

            value = getattr(report.metrics, metric_name)(data_source=source)

            if isinstance(value, Number):
                # Ignore list[value] (multi-output)
                # Ignore {label: value} (multi-class)
                metrics.append(
                    {
                        "name": metric_name,
                        "value": value,
                        "data_source": source,
                        "greater_is_better": None,  # how to get efficiently?
                        "position": None,  # int: to plot, None: to ignore
                    }
                )

        return {
            "estimator_class_name": estimator.__class__.__name__,
            "estimator_hyper_params": {
                k: v for k, v in estimator.get_params().items() if is_primitive(v)
            },
            "dataset_fingerprint": joblib_hash(
                (report.X_train, report.y_train, report.X_test, report.y_test)
            ),
            "ml_task": report.ml_task,  # modify hub's allowed values
            "metrics": metrics,
        }

    @property
    def __representation__(self) -> dict[str, dict]:
        from . import object_to_item

        report = self.__raw__
        representation = []

        for plot_name, source in itertools.product(PLOTS, ("test", "train")):
            if hasattr(report.metrics, plot_name):
                figure, ax = subplots()
                display = getattr(report.metrics, plot_name)(data_source=source)

                # construct the ``matplotlib`` figure from the ``skore`` display object
                display.plot(ax)

                item = object_to_item(figure)
                representation.append(
                    {
                        "key": plot_name,
                        "data_source": source,
                        **item.__parameters__,
                        **item.__representation__,
                    }
                )

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
