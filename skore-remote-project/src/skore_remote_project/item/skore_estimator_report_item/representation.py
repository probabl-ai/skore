from inspect import signature
from operator import attrgetter

from matplotlib.pyplot import subplots
from sklearn.utils import estimator_html_repr

from ..matplotlib_figure_item import MatplotlibFigureItem
from ..media_item import MediaItem
from ..pandas_dataframe_item import PandasDataFrameItem


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
