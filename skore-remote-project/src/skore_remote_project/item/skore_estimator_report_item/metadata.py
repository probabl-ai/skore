from contextlib import suppress
from inspect import getmembers, ismethod

from joblib import hash


def metadata(function):
    function.metadata = ...
    return function


class Metadata:
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
        return hash(
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
