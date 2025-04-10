from contextlib import suppress
from inspect import getmembers, ismethod

from sklearn.utils import estimator_html_repr

from ..matplotlib_figure_item import MatplotlibFigureItem
from ..media_item import MediaItem
from ..pandas_dataframe_item import PandasDataFrameItem


class Representation:
    def representation(function):
        function.representation = ...
        return function

    def __init__(self, report):
        self.report = report

    @representation
    def precision_recall_train(self):
        with suppress(AttributeError):
            pr = self.report.feature_importance.precision_recall(data_source="train")
            item = MatplotlibFigureItem.factory(pr.plot().figure_)
            item_representation = item.__representation__

            return {
                "category": "performance",
                "key": "precision_recall_train",
                "attributes": {"data_source": "train"},
                **item_representation,
            }

    @representation
    def precision_recall_test(self):
        with suppress(AttributeError):
            pr = self.report.feature_importance.precision_recall(data_source="test")
            item = MatplotlibFigureItem.factory(pr.plot().figure_)
            item_representation = item.__representation__

            return {
                "category": "performance",
                "key": "precision_recall_test",
                "attributes": {"data_source": "test"},
                **item_representation,
            }

    @representation
    def prediction_error_train(self):
        with suppress(AttributeError):
            pe = self.report.feature_importance.prediction_error(data_source="train")
            item = MatplotlibFigureItem.factory(pe.plot().figure_)
            item_representation = item.__representation__

            return {
                "category": "performance",
                "key": "prediction_error_train",
                "attributes": {"data_source": "train"},
                **item_representation,
            }

    @representation
    def prediction_error_test(self):
        with suppress(AttributeError):
            pe = self.report.feature_importance.prediction_error(data_source="test")
            item = MatplotlibFigureItem.factory(pe.plot().figure_)
            item_representation = item.__representation__

            return {
                "category": "performance",
                "key": "prediction_error_test",
                "attributes": {"data_source": "test"},
                **item_representation,
            }

    @representation
    def roc_train(self):
        with suppress(AttributeError):
            roc = self.report.feature_importance.roc(data_source="train")
            item = MatplotlibFigureItem.factory(roc.plot().figure_)
            item_representation = item.__representation__

            return {
                "category": "performance",
                "key": "roc_train",
                "attributes": {"data_source": "train"},
                **item_representation,
            }

    @representation
    def roc_test(self):
        with suppress(AttributeError):
            roc = self.report.feature_importance.roc(data_source="test")
            item = MatplotlibFigureItem.factory(roc.plot().figure_)
            item_representation = item.__representation__

            return {
                "category": "performance",
                "key": "roc_test",
                "attributes": {"data_source": "test"},
                **item_representation,
            }

    @representation
    def permutation_train(self):
        p = self.report.feature_importance.permutation(data_source="train")
        item = PandasDataFrameItem.factory(p)
        item_representation = item.__representation__

        return {
            "category": "feature_importance",
            "key": "permutation_train",
            "attributes": {"data_source": "train", "method": "permutation"},
            **item_representation,
        }

    @representation
    def permutation_test(self):
        p = self.report.feature_importance.permutation(data_source="test")
        item = PandasDataFrameItem.factory(p)
        item_representation = item.__representation__

        return {
            "category": "feature_importance",
            "key": "permutation_test",
            "attributes": {"data_source": "test", "method": "permutation"},
            **item_representation,
        }

    @representation
    def mean_decrease_impurity(self):
        with suppress(AttributeError):
            mdi = self.report.feature_importance.mean_decrease_impurity()
            item = PandasDataFrameItem.factory(mdi)
            item_representation = item.__representation__

            return {
                "category": "feature_importance",
                "key": "mean_decrease_impurity",
                "attributes": {"method": "mean_decrease_impurity"},
                **item_representation,
            }

    @representation
    def coefficients(self):
        with suppress(AttributeError):
            c = self.report.feature_importance.coefficients()
            item = PandasDataFrameItem.factory(c)
            item_representation = item.__representation__

            return {
                "category": "feature_importance",
                "key": "coefficients",
                "attributes": {"method": "coefficients"},
                **item_representation,
            }

    @representation
    def estimator_html_repr(self):
        e = estimator_html_repr(self.report.estimator_)
        item = MediaItem.factory(e, media_type="text/html")
        item_representation = item.__representation__

        return {
            "category": "model",
            "key": "estimator_html_repr",
            "attributes": None,
            **item_representation,
        }

    def __iter__(self):
        for key, method in getmembers(self):
            if ismethod(method) and hasattr(method, "representation"):
                if (value := method()) is not None:
                    yield (key, value)
