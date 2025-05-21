"""
SkoreEstimatorReportItem.

This module defines the ``SkoreEstimatorReportItem`` class used to serialize instances
of ``skore.EstimatorReport``, using binary protocols.
"""

from __future__ import annotations

from contextlib import suppress
from inspect import getmembers, ismethod, signature
from math import isfinite
from operator import attrgetter
from typing import TYPE_CHECKING

from .item import ItemTypeError, lazy_is_instance, switch_mpl_backend
from .matplotlib_figure_item import MatplotlibFigureItem
from .media_item import MediaItem
from .pandas_dataframe_item import PandasDataFrameItem
from .pickle_item import PickleItem

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, Literal, TypedDict, Union

    from skore.sklearn import EstimatorReport

    class MetadataFunction:  # noqa: D101
        metadata: Any

    class Metric(TypedDict):  # noqa: D101
        name: str
        verbose_name: Union[str, None]
        value: float
        data_source: Union[str, None]
        greater_is_better: Union[bool, None]
        position: Union[int, None]

    class Representation(TypedDict):  # noqa: D101
        key: str
        verbose_name: Union[str, None]
        category: Literal["performance", "feature_importance", "model"]
        attributes: dict
        representation: dict
        parameters: dict


def cast_to_float(value: Any) -> Union[float, None]:
    """Cast value to float."""
    with suppress(TypeError):
        if (value := float(value)) and isfinite(value):
            return value

    return None


def metadata_function(function: Any) -> MetadataFunction:
    """
    Decorate function that has to be annotated as ``metadata``.

    Annotated functions as ``medatata`` are dynamically retrieved and called at runtime
    to compute a snapshot of all the available metadata of a report.

    Notes
    -----
    This decorator uses the `function attributes <https://peps.python.org/pep-0232/>`_.
    """
    function.metadata = ...
    return function


class Metadata:
    """
    Compute report's metadata to exchange with ``skore hub``.

    Notes
    -----
    To add a new metadata to compute, you have to add a new method decorated as
    ``metadata``. Its name is used as identifier.
    """

    def __init__(self, report: EstimatorReport):
        """
        Initialize a ``Metadata``.

        Parameters
        ----------
        report : ``skore.EstimatorReport``
            The report from which computing the metadata.
        """
        self.report = report

    @metadata_function
    def estimator_class_name(self) -> str:
        """Return the name of the report's estimator."""
        return self.report.estimator_name_

    @metadata_function
    def estimator_hyper_params(self) -> dict:
        """DeprecationWarning: send empty dictionary to not break the hub API."""
        return {}

    @metadata_function
    def dataset_fingerprint(self) -> str:
        """Return the hash of the targets in the test-set."""
        import joblib

        return joblib.hash(self.report.y_test)

    @metadata_function
    def ml_task(self) -> str:
        """Return the type of ML task covered by the report."""
        return self.report._ml_task

    @metadata_function
    def metrics(self) -> list[Metric]:
        """
        Return the list of scalar metrics that can be computed from the report.

        Notes
        -----
        Unavailable metrics are automatically filtered out.

        All metrics whose value is not a scalar are currently ignored:
        - ignore ``list[float]`` for multi-output ML task,
        - ignore ``dict[str: float]`` for multi-classes ML task.

        The position field is used to drive the HUB's parallel coordinates plot:
        - int [0, inf[, to be displayed at the position,
        - None, not to be displayed.
        """

        def metric(
            name: str,
            verbose_name: str,
            data_source: str,
            greater_is_better: bool,
            position: Union[int, None],
            /,
        ) -> Union[Metric, None]:
            if hasattr(self.report.metrics, name):
                value = getattr(self.report.metrics, name)(data_source=data_source)

                if (value := cast_to_float(value)) is not None:
                    return {
                        "name": name,
                        "verbose_name": verbose_name,
                        "value": value,
                        "data_source": data_source,
                        "greater_is_better": greater_is_better,
                        "position": position,
                    }

            return None

        def timing(
            name: str,
            verbose_name: str,
            data_source: Union[str, None],
            greater_is_better: bool,
            position: Union[int, None],
            /,
        ) -> Union[Metric, None]:
            timings = self.report.metrics.timings()
            value = timings.get(
                name if name != "predict_time" else f"{name}_{data_source}"
            )

            if (value := cast_to_float(value)) is not None:
                return {
                    "name": name,
                    "verbose_name": verbose_name,
                    "value": value,
                    "data_source": data_source,
                    "greater_is_better": greater_is_better,
                    "position": position,
                }

            return None

        return list(
            filter(
                None,
                (
                    metric("accuracy", "Accuracy", "train", True, None),
                    metric("accuracy", "Accuracy", "test", True, None),
                    metric("brier_score", "Brier score", "train", False, None),
                    metric("brier_score", "Brier score", "test", False, None),
                    metric("log_loss", "Log loss", "train", False, 4),
                    metric("log_loss", "Log loss", "test", False, 4),
                    metric("precision", "Precision", "train", True, None),
                    metric("precision", "Precision", "test", True, None),
                    metric("r2", "R²", "train", True, None),
                    metric("r2", "R²", "test", True, None),
                    metric("recall", "Recall", "train", True, None),
                    metric("recall", "Recall", "test", True, None),
                    metric("rmse", "RMSE", "train", False, 3),
                    metric("rmse", "RMSE", "test", False, 3),
                    metric("roc_auc", "ROC AUC", "train", True, 3),
                    metric("roc_auc", "ROC AUC", "test", True, 3),
                    # timings must be calculated last
                    timing("fit_time", "Fit time (s)", None, False, 1),
                    timing("predict_time", "Predict time (s)", "train", False, 2),
                    timing("predict_time", "Predict time (s)", "test", False, 2),
                ),
            )
        )

    def __iter__(self) -> Generator[tuple[str, Any]]:
        """
        Dynamically generate metadata by calling all ``metadata`` methods.

        Notes
        -----
        Null metadata are automatically filtered out.
        """
        for key, method in getmembers(self):
            if (
                ismethod(method)
                and hasattr(method, "metadata")
                and ((value := method()) is not None)
            ):
                yield (key, value)


class Representations:
    """Compute report's representation to exchange with ``skore hub``."""

    def __init__(self, report):
        """
        Initialize a ``Metadata``.

        Parameters
        ----------
        report : ``skore.EstimatorReport``
            The report from which computing the metadata.
        """
        self.report = report

    def mpl(
        self,
        name,
        verbose_name,
        category,
        **kwargs,
    ) -> Union[Representation, None]:
        """Return sub-representation made of ``matplotlib`` figures."""
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

            with switch_mpl_backend():
                display.plot()

            item = MatplotlibFigureItem.factory(display.figure_)

            return {
                "key": name.split(".")[-1],
                "verbose_name": verbose_name,
                "category": category,
                "attributes": kwargs,
                "parameters": {},
                "representation": item.__representation__["representation"],
            }

    def pd(self, name, verbose_name, category, **kwargs) -> Union[Representation, None]:
        """Return sub-representation made of ``pandas`` dataframes."""
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
                "verbose_name": verbose_name,
                "category": category,
                "attributes": kwargs,
                "parameters": {},
                "representation": item.__representation__["representation"],
            }

    def estimator_html_repr(self) -> Representation:
        """Return ``sklearn`` HTML representation of the report's estimator."""
        import sklearn.utils

        item = MediaItem.factory(
            sklearn.utils.estimator_html_repr(self.report.estimator_),
            media_type="text/html",
        )

        return {
            "key": "estimator_html_repr",
            "verbose_name": None,
            "category": "model",
            "attributes": {},
            "parameters": {},
            "representation": item.__representation__["representation"],
        }

    def __iter__(self) -> Generator[Representation]:
        """
        Dynamically generate representation, with report's parameters combinations.

        Notes
        -----
        Null representations are automatically filtered out.
        """
        yield from filter(
            None,
            (
                self.mpl(
                    "metrics.precision_recall",
                    "Precision Recall",
                    "performance",
                    data_source="train",
                ),
                self.mpl(
                    "metrics.precision_recall",
                    "Precision Recall",
                    "performance",
                    data_source="test",
                ),
                self.mpl(
                    "metrics.prediction_error",
                    "Prediction error",
                    "performance",
                    data_source="train",
                ),
                self.mpl(
                    "metrics.prediction_error",
                    "Prediction error",
                    "performance",
                    data_source="test",
                ),
                self.mpl("metrics.roc", "ROC", "performance", data_source="train"),
                self.mpl("metrics.roc", "ROC", "performance", data_source="test"),
                self.pd(
                    "feature_importance.permutation",
                    "Feature importance - Permutation",
                    "feature_importance",
                    data_source="train",
                    method="permutation",
                ),
                self.pd(
                    "feature_importance.permutation",
                    "Feature importance - Permutation",
                    "feature_importance",
                    data_source="test",
                    method="permutation",
                ),
                self.pd(
                    "feature_importance.mean_decrease_impurity",
                    "Feature importance - Mean Decrease Impurity (MDI)",
                    "feature_importance",
                    method="mean_decrease_impurity",
                ),
                self.pd(
                    "feature_importance.coefficients",
                    "Feature importance - Coefficients",
                    "feature_importance",
                    method="coefficients",
                ),
                self.estimator_html_repr(),
            ),
        )


class SkoreEstimatorReportItem(PickleItem):
    """Serialize instances of ``skore.EstimatorReport``, using binary protocols."""

    @property
    def __metadata__(self) -> dict[str, Any]:
        """Get the metadata of the ``SkoreEstimatorReportItem`` instance."""
        return dict(Metadata(self.__raw__))

    @property
    def __representation__(self) -> dict[str, list[Representation]]:
        """Get the representation of the ``SkoreEstimatorReportItem`` instance."""
        return {"related_items": list(Representations(self.__raw__))}

    @classmethod
    def factory(cls, value: EstimatorReport, /) -> SkoreEstimatorReportItem:
        """
        Create a new ``SkoreEstimatorReportItem``.

        Create a new ``SkoreEstimatorReportItem`` from an instance of
        ``skore.EstimatorReport``.

        Parameters
        ----------
        value : ``skore.EstimatorReport``
            The value to serialize.

        Returns
        -------
        SkoreEstimatorReportItem
            A new ``SkoreEstimatorReportItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` is not an instance of ``skore.EstimatorReport``.
        """
        if lazy_is_instance(value, "skore.sklearn._estimator.report.EstimatorReport"):
            return super().factory(value)

        raise ItemTypeError(f"Type '{value.__class__}' is not supported.")
