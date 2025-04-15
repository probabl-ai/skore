from __future__ import annotations

from contextlib import suppress
from inspect import getmembers, ismethod, signature
from operator import attrgetter
from typing import TYPE_CHECKING

from joblib import hash
from matplotlib.pyplot import subplots
from sklearn.utils import estimator_html_repr

from .item import ItemTypeError, lazy_is_instance
from .matplotlib_figure_item import MatplotlibFigureItem
from .media_item import MediaItem
from .pandas_dataframe_item import PandasDataFrameItem
from .pickle_item import PickleItem

if TYPE_CHECKING:
    from skore import EstimatorReport


class Metadata:
    def metadata(function):
        function.metadata = ...
        return function

    def __init__(self, report):
        self.report = report

    @metadata
    def estimator_class_name(self):
        return self.report.estimator_name_

    @metadata
    def estimator_hyper_params(self):
        return {
            key: value
            for key, value in self.report.estimator_.get_params().items()
            if isinstance(value, (type(None), bool, float, int, str))
        }

    @metadata
    def dataset_fingerprint(self):
        return hash(self.report.y_test)

    @metadata
    def ml_task(self):
        return self.report.ml_task

    @metadata
    def metrics(self):
        #
        # Value:
        # - ignore list[value] (multi-output)
        # - ignore {label: value} (multi-class)
        #
        # Position: int (to display in parallel coordinates plot) | None (to ignore)
        #

        def scalar(name, data_source, greater_is_better, position, /):
            with suppress(AttributeError, TypeError):
                function = getattr(self.report.metrics, name)
                value = float(function(data_source=data_source))

                return {
                    "name": name,
                    "value": value,
                    "data_source": data_source,
                    "greater_is_better": greater_is_better,
                    "position": position,
                }

        def timing(name, data_source, position, /):
            with suppress(KeyError):
                return {
                    "name": name,
                    "value": self.report.metrics.timings()[name],
                    "data_source": data_source,
                    "greater_is_better": False,
                    "position": position,
                }

        return list(
            filter(
                lambda value: value is not None,
                (
                    scalar("accuracy", "train", True, None),
                    scalar("accuracy", "test", True, None),
                    scalar("brier_score", "train", False, None),
                    scalar("brier_score", "test", False, None),
                    scalar("log_loss", "train", False, 4),
                    scalar("log_loss", "test", False, 4),
                    scalar("precision", "train", True, None),
                    scalar("precision", "test", True, None),
                    scalar("r2", "train", True, None),
                    scalar("r2", "test", True, None),
                    scalar("recall", "train", True, None),
                    scalar("recall", "test", True, None),
                    scalar("rmse", "train", False, 3),
                    scalar("rmse", "test", False, 3),
                    scalar("roc_auc", "train", True, 3),
                    scalar("roc_auc", "test", True, 3),
                    timing("fit_time", None, 1),
                    timing("predict_time_test", "test", 2),
                    timing("predict_time_train", "train", 2),
                ),
            )
        )

    def __iter__(self):
        for key, method in getmembers(self):
            if (
                ismethod(method)
                and hasattr(method, "metadata")
                and ((value := method()) is not None)
            ):
                yield (key, value)


class Representation:
    def __init__(self, report):
        self.report = report

    def mpl(self, name, category, **kwargs):
        try:
            function = attrgetter(name)(self.report)
        except AttributeError:
            return None
        else:
            function_parameters = signature(function).parameters
            function_kwargs = {
                k: v for k, v in kwargs.items() if k in function_parameters
            }
            display = function(**function_kwargs)
            figure, ax = subplots()
            display.plot(ax)

            item = MatplotlibFigureItem.factory(figure)

            return {
                "key": name.split(".")[-1],
                "category": category,
                "attributes": kwargs,
                **item.__representation__,
                **item.__parameters__,
            }

    def pd(self, name, category, **kwargs):
        try:
            function = attrgetter(name)(self.report)
        except AttributeError:
            return None
        else:
            function_parameters = signature(function).parameters
            function_kwargs = {
                k: v for k, v in kwargs.items() if k in function_parameters
            }
            dataframe = function(**function_kwargs)
            item = PandasDataFrameItem.factory(dataframe)

            return {
                "key": name.split(".")[-1],
                "category": category,
                "attributes": kwargs,
                **item.__representation__,
                **item.__parameters__,
            }

    def estimator_html_repr(self):
        e = estimator_html_repr(self.report.estimator_)
        item = MediaItem.factory(e, media_type="text/html")

        return {
            "key": "estimator_html_repr",
            "category": "model",
            "attributes": {},
            **item.__representation__,
            **item.__parameters__,
        }

    def __iter__(self):
        # fmt: off
        yield from filter(
            lambda value: value is not None,
            (
                self.mpl("metrics.precision_recall", "performance", data_source="train"),  # noqa: E501
                self.mpl("metrics.precision_recall", "performance", data_source="test"),  # noqa: E501
                self.mpl("metrics.prediction_error", "performance", data_source="train"),  # noqa: E501
                self.mpl("metrics.prediction_error", "performance", data_source="test"),  # noqa: E501
                self.mpl("metrics.roc", "performance", data_source="train"),  # noqa: E501
                self.mpl("metrics.roc", "performance", data_source="test"),  # noqa: E501
                self.pd("feature_importance.permutation", "feature_importance", data_source="train", method="permutation"),  # noqa: E501
                self.pd("feature_importance.permutation", "feature_importance", data_source="test", method="permutation"),  # noqa: E501
                self.pd("feature_importance.mean_decrease_impurity", "feature_importance", method="mean_decrease_impurity"),  # noqa: E501
                self.pd("feature_importance.coefficients", "feature_importance", method="coefficients"),  # noqa: E501
                self.estimator_html_repr(),  # noqa: E501
            ),
        )
        # fmt: off


class SkoreEstimatorReportItem(PickleItem):
    @property
    def __metadata__(self) -> dict[str, float]:
        return dict(Metadata(self.__raw__))

    @property
    def __representation__(self) -> dict[str, dict]:
        return {"related_items": list(Representation(self.__raw__))}

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
