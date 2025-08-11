from functools import cached_property, reduce
from inspect import signature
from io import BytesIO
from typing import ClassVar, Literal

from matplotlib import pyplot as plt
from pydantic import Field, computed_field
from skore import CrossValidationReport, EstimatorReport

from .media import Media, Representation


class Performance(Media):
    report: EstimatorReport | CrossValidationReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str]
    category: Literal["performance"] = "performance"

    @computed_field
    @cached_property
    def representation(self) -> Representation | None:
        from skore_hub_project import bytes_to_b64_str, switch_mpl_backend

        try:
            function = reduce(getattr, self.accessor.split("."), self.report)
        except AttributeError:
            return None

        function_parameters = signature(function).parameters
        function_kwargs = {
            k: v for k, v in self.attributes.items() if k in function_parameters
        }

        display = function(**function_kwargs)

        with switch_mpl_backend(), BytesIO() as stream:
            display.plot()
            display.figure_.savefig(stream, format="svg", bbox_inches="tight")

            figure_bytes = stream.getvalue()
            figure_b64_str = bytes_to_b64_str(figure_bytes)

            plt.close(display.figure_)

        return Representation(
            media_type="image/svg+xml;base64",
            value=figure_b64_str,
        )


class PrecisionRecall(Performance):
    accessor: ClassVar[Literal["metrics.precision_recall"]] = "metrics.precision_recall"
    key: Literal["precision_recall"] = "precision_recall"
    verbose_name: Literal["Precision Recall"] = "Precision Recall"


class PrecisionRecallTrain(PrecisionRecall):
    attributes: Literal[{"data_source": "train"}] = {"data_source": "train"}


class PrecisionRecallTest(PrecisionRecall):
    attributes: Literal[{"data_source": "test"}] = {"data_source": "test"}


class PredictionError(Performance):
    accessor: ClassVar[Literal["metrics.prediction_error"]] = "metrics.prediction_error"
    key: Literal["prediction_error"] = "prediction_error"
    verbose_name: Literal["Prediction error"] = "Prediction error"


class PredictionErrorTrain(PredictionError):
    attributes: Literal[{"data_source": "train"}] = {"data_source": "train"}


class PredictionErrorTest(PredictionError):
    attributes: Literal[{"data_source": "test"}] = {"data_source": "test"}


class Roc(Performance):
    accessor: ClassVar[Literal["metrics.roc"]] = "metrics.roc"
    key: Literal["roc"] = "roc"
    verbose_name: Literal["ROC"] = "ROC"


class RocTrain(Roc):
    attributes: Literal[{"data_source": "train"}] = {"data_source": "train"}


class RocTest(Roc):
    attributes: Literal[{"data_source": "test"}] = {"data_source": "test"}
