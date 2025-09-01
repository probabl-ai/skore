"""Class definition of the payload used to send a performance media to ``hub``."""

from collections.abc import Callable
from functools import cached_property, reduce
from inspect import signature
from io import BytesIO
from typing import ClassVar, Literal, cast

from matplotlib import pyplot as plt
from pydantic import Field, computed_field
from skore import CrossValidationReport, EstimatorReport

from .media import Media, Representation


class Performance(Media):  # noqa: D101
    report: EstimatorReport | CrossValidationReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str]
    category: Literal["performance"] = "performance"

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def representation(self) -> Representation | None:  # noqa: D102
        from skore_hub_project import bytes_to_b64_str, switch_mpl_backend

        try:
            function = cast(
                Callable,
                reduce(getattr, self.accessor.split("."), self.report),
            )
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


class PrecisionRecall(Performance):  # noqa: D101
    accessor: ClassVar[str] = "metrics.precision_recall"
    key: str = "precision_recall"
    verbose_name: str = "Precision Recall"


class PrecisionRecallTrain(PrecisionRecall):  # noqa: D101
    attributes: dict = {"data_source": "train"}


class PrecisionRecallTest(PrecisionRecall):  # noqa: D101
    attributes: dict = {"data_source": "test"}


class PredictionError(Performance):  # noqa: D101
    accessor: ClassVar[str] = "metrics.prediction_error"
    key: str = "prediction_error"
    verbose_name: str = "Prediction error"


class PredictionErrorTrain(PredictionError):  # noqa: D101
    attributes: dict = {"data_source": "train"}


class PredictionErrorTest(PredictionError):  # noqa: D101
    attributes: dict = {"data_source": "test"}


class Roc(Performance):  # noqa: D101
    accessor: ClassVar[str] = "metrics.roc"
    key: str = "roc"
    verbose_name: str = "ROC"


class RocTrain(Roc):  # noqa: D101
    attributes: dict = {"data_source": "train"}


class RocTest(Roc):  # noqa: D101
    attributes: dict = {"data_source": "test"}
