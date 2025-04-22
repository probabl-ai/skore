from __future__ import annotations

from contextlib import suppress
from inspect import getmembers, ismethod, signature
from operator import attrgetter
from typing import TYPE_CHECKING

from joblib import hash as joblib_hash
from matplotlib.pyplot import close, subplots
from sklearn.utils import estimator_html_repr

from .item import ItemTypeError, lazy_is_instance
from .matplotlib_figure_item import MatplotlibFigureItem
from .media_item import MediaItem
from .pandas_dataframe_item import PandasDataFrameItem
from .pickle_item import PickleItem

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, Literal, Optional, TypedDict, Union

    from skore import EstimatorReport

    class MetadataFunction:
        metadata: Any

    class Metric(TypedDict):
        name: str
        value: float
        data_source: Optional[str]
        greater_is_better: Optional[bool]
        position: Optional[int]

    class Representation(TypedDict):
        key: str
        category: Literal["performance", "feature_importance", "model"]
        attributes: dict
        representation: dict
        parameters: dict


def metadata_function(function: Any) -> MetadataFunction:
    """
    Decorate function that has to be marked as ``metadata``.

    Notes
    -----
    Marked functions as ``medatata`` are dynamically retrieved and called at runtime
    to compute a snapshot of all the available metadata of a reports.
    """
    function.metadata = ...
    return function


class Metadata:
    def __init__(self, report: EstimatorReport):
        self.report = report

    @metadata_function
    def estimator_class_name(self) -> str:
        """Return the name of the report's estimator."""
        return self.report.estimator_name_

    @metadata_function
    def estimator_hyper_params(self) -> dict[str, Union[None, bool, float, int, str]]:
        """Return the primitive hyper parameters of the report's estimator."""
        return {
            key: value
            for key, value in self.report.estimator_.get_params().items()
            if isinstance(value, (type(None), bool, float, int, str))
        }

    @metadata_function
    def dataset_fingerprint(self) -> str:
        """Return the hash of the targets in the test-set."""
        return joblib_hash(self.report.y_test)

    @metadata_function
    def ml_task(self) -> str:
        """Return the type of ML task covered by the report."""
        return self.report.ml_task

    @metadata_function
    def metrics(self) -> list[Metric]:
        """
        Return the list of scalar metrics that can be computed from the report.

        Notes
        -----
        All metrics whose value is not a scalar are currently ignored:
        - ignore ``list[float]`` for multi-output ML task,
        - ignore ``dict[str: float]`` for multi-classes ML task.

        The position field is used to drive the HUB's parallel coordinates plot:
        - int [0, inf[ to be displayed at the position,
        - None not to be displayed.
        """

        def metric(
            name: str,
            data_source: Optional[str] = None,
            greater_is_better: Optional[bool] = None,
            position: Optional[int] = None,
            /,
        ) -> Union[Metric, None]:
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
            return None

        def timing(
            name: str,
            data_source: Optional[str] = None,
            position: Optional[int] = None,
            /,
        ) -> Union[Metric, None]:
            with suppress(KeyError, TypeError):
                return {
                    "name": name,
                    "value": float(self.report.metrics.timings()[name]),
                    "data_source": data_source,
                    "greater_is_better": False,
                    "position": position,
                }
            return None

        return list(
            filter(
                None,
                (
                    metric("accuracy", "train", True, None),
                    metric("accuracy", "test", True, None),
                    metric("brier_score", "train", False, None),
                    metric("brier_score", "test", False, None),
                    metric("log_loss", "train", False, 4),
                    metric("log_loss", "test", False, 4),
                    metric("precision", "train", True, None),
                    metric("precision", "test", True, None),
                    metric("r2", "train", True, None),
                    metric("r2", "test", True, None),
                    metric("recall", "train", True, None),
                    metric("recall", "test", True, None),
                    metric("rmse", "train", False, 3),
                    metric("rmse", "test", False, 3),
                    metric("roc_auc", "train", True, 3),
                    metric("roc_auc", "test", True, 3),
                    timing("fit_time", None, 1),
                    timing("predict_time_test", "test", 2),
                    timing("predict_time_train", "train", 2),
                ),
            )
        )

    def __iter__(self) -> Generator[tuple[str, Any]]:
        for key, method in getmembers(self):
            if (
                ismethod(method)
                and hasattr(method, "metadata")
                and ((value := method()) is not None)
            ):
                yield (key, value)


class Representations:
    def __init__(self, report):
        self.report = report

    def mpl(self, name, category, **kwargs) -> Union[Representation, None]:
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
            close(figure)

            item = MatplotlibFigureItem.factory(figure)

            return {
                "key": name.split(".")[-1],
                "category": category,
                "attributes": kwargs,
                "parameters": {},
                "representation": item.__representation__["representation"],
            }

    def pd(self, name, category, **kwargs) -> Union[Representation, None]:
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
                "parameters": {},
                "representation": item.__representation__["representation"],
            }

    def estimator_html_repr(self) -> Representation:
        e = estimator_html_repr(self.report.estimator_)
        item = MediaItem.factory(e, media_type="text/html")

        return {
            "key": "estimator_html_repr",
            "category": "model",
            "attributes": {},
            "parameters": {},
            "representation": item.__representation__["representation"],
        }

    def __iter__(self) -> Generator[Representation]:
        # fmt: off
        yield from filter(
            None,
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
    def __metadata__(self) -> dict[str, Any]:
        return dict(Metadata(self.__raw__))

    @property
    def __representation__(self) -> dict[str, list[Representation]]:
        return {"related_items": list(Representations(self.__raw__))}

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
