from contextlib import suppress
from functools import partial
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

        def scalar(name, data_source, greater_is_better=None, position=None, /):
            with suppress(AttributeError, TypeError):
                metric = getattr(self.report.metrics, name)
                value = float(metric(data_source=data_source))

                return {
                    "name": name,
                    "value": value,
                    "data_source": data_source,
                    "greater_is_better": greater_is_better,
                    "position": position,
                }

        # fmt: off
        accuracy_train = metadata(partial(scalar, "accuracy", "train", True))  # noqa: E501, F841
        accuracy_test = metadata(partial(scalar, "accuracy", "test", True))  # noqa: E501, F841
        brier_score_train = metadata(partial(scalar, "brier_score", "train", False))  # noqa: E501, F841
        brier_score_test = metadata(partial(scalar, "brier_score", "test", False))  # noqa: E501, F841
        log_loss_train = metadata(partial(scalar, "log_loss", "train", False))  # noqa: E501, F841
        log_loss_test = metadata(partial(scalar, "log_loss", "test", False))  # noqa: E501, F841
        precision_train = metadata(partial(scalar, "precision", "train", True))  # noqa: E501, F841
        precision_test = metadata(partial(scalar, "precision", "test", True))  # noqa: E501, F841
        r2_train = metadata(partial(scalar, "r2", "train", True))  # noqa: E501, F841
        r2_test = metadata(partial(scalar, "r2", "test", True))  # noqa: E501, F841
        recall_train = metadata(partial(scalar, "recall", "train", True))  # noqa: E501, F841
        recall_test = metadata(partial(scalar, "recall", "test", True))  # noqa: E501, F841
        rmse_train = metadata(partial(scalar, "rmse", "train", False))  # noqa: E501, F841
        rmse_test = metadata(partial(scalar, "rmse", "test", False))  # noqa: E501, F841
        roc_auc_train = metadata(partial(scalar, "roc_auc", "train", True))  # noqa: E501, F841
        roc_auc_test = metadata(partial(scalar, "roc_auc", "test", True))  # noqa: E501, F841
        # fmt: on

        @metadata
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
                (
                    metric()
                    for metric in locals().values()
                    if callable(metric) and hasattr(metric, "metadata")
                ),
            )
        )

    def __iter__(self):
        for key, method in getmembers(self):
            if ismethod(method) and hasattr(method, "metadata"):
                if (value := method()) is not None:
                    yield (key, value)
