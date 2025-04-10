from contextlib import suppress
from inspect import getmembers, ismethod

from joblib import hash as joblib_hash


class Metadata:
    def metadata(function):
        function.metadata = ...
        return function

    def __init__(self, report):
        self.report = report

    @metadata
    def estimator_class_name(self):
        return self.report.estimator_.__class__.__name__

    @metadata
    def estimator_hyper_params(self):
        return {
            key: value
            for key, value in self.report.estimator_.get_params().items()
            if isinstance(value, (type(None), bool, float, int, str))
        }

    @metadata
    def dataset_fingerprint(self):
        return joblib_hash(
            (
                self.report.X_train,
                self.report.y_train,
                self.report.X_test,
                self.report.y_test,
            )
        )

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

        def accuracy_train():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "accuracy_train",
                    "value": float(self.report.metrics.accuracy(data_source="train")),
                    "data_source": "train",
                    "greater_is_better": True,
                    "position": None,
                }

        def accuracy_test():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "accuracy_test",
                    "value": float(self.report.metrics.accuracy(data_source="test")),
                    "data_source": "test",
                    "greater_is_better": True,
                    "position": None,
                }

        def brier_score_train():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "brier_score_train",
                    "value": float(
                        self.report.metrics.brier_score(data_source="train")
                    ),
                    "data_source": "train",
                    "greater_is_better": False,
                    "position": None,
                }

        def brier_score_test():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "brier_score_test",
                    "value": float(self.report.metrics.brier_score(data_source="test")),
                    "data_source": "test",
                    "greater_is_better": False,
                    "position": None,
                }

        def log_loss_train():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "log_loss_train",
                    "value": float(self.report.metrics.log_loss(data_source="train")),
                    "data_source": "train",
                    "greater_is_better": False,
                    "position": None,
                }

        def log_loss_test():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "log_loss_test",
                    "value": float(self.report.metrics.log_loss(data_source="test")),
                    "data_source": "test",
                    "greater_is_better": False,
                    "position": None,
                }

        def precision_train():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "precision_train",
                    "value": float(self.report.metrics.precision(data_source="train")),
                    "data_source": "train",
                    "greater_is_better": True,
                    "position": None,
                }

        def precision_test():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "precision_test",
                    "value": float(self.report.metrics.precision(data_source="test")),
                    "data_source": "test",
                    "greater_is_better": True,
                    "position": None,
                }

        def r2_train():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "r2_train",
                    "value": float(self.report.metrics.r2(data_source="train")),
                    "data_source": "train",
                    "greater_is_better": True,
                    "position": None,
                }

        def r2_test():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "r2_test",
                    "value": float(self.report.metrics.r2(data_source="test")),
                    "data_source": "test",
                    "greater_is_better": True,
                    "position": None,
                }

        def recall_train():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "recall_train",
                    "value": float(self.report.metrics.recall(data_source="train")),
                    "data_source": "train",
                    "greater_is_better": True,
                    "position": None,
                }

        def recall_test():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "recall_test",
                    "value": float(self.report.metrics.recall(data_source="test")),
                    "data_source": "test",
                    "greater_is_better": True,
                    "position": None,
                }

        def rmse_train():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "rmse_train",
                    "value": float(self.report.metrics.rmse(data_source="train")),
                    "data_source": "train",
                    "greater_is_better": False,
                    "position": None,
                }

        def rmse_test():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "rmse_test",
                    "value": float(self.report.metrics.rmse(data_source="test")),
                    "data_source": "test",
                    "greater_is_better": False,
                    "position": None,
                }

        def roc_auc_train():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "roc_auc_train",
                    "value": float(self.report.metrics.roc_auc(data_source="train")),
                    "data_source": "train",
                    "greater_is_better": True,
                    "position": None,
                }

        def roc_auc_test():
            with suppress(AttributeError, TypeError):
                return {
                    "name": "roc_auc_test",
                    "value": float(self.report.metrics.roc_auc(data_source="test")),
                    "data_source": "test",
                    "greater_is_better": True,
                    "position": None,
                }

        def fit_time():
            with suppress(KeyError):
                return {
                    "name": "fit_time",
                    "value": self.report.metrics.timings()["fit_time"],
                    "greater_is_better": False,
                    "position": None,
                }

        return list(
            filter(
                lambda value: value is not None,
                (metric() for metric in locals().values() if callable(metric)),
            )
        )

    def __iter__(self):
        for name, member in getmembers(self):
            if ismethod(member) and hasattr(member, "metadata"):
                yield (name, member())
