"""Class definition of the payload used to send feature importance medias to ``hub``."""

from abc import ABC
from collections.abc import Callable
from functools import reduce
from typing import ClassVar, Literal, cast

from pandas import DataFrame
from pydantic import Field

from skore_hub_project.artifact.media.media import Media
from skore_hub_project.artifact.serializer import JsonSerializer
from skore_hub_project.protocol import EstimatorReport


class FeatureImportance(Media, ABC):  # noqa: D101
    accessor: ClassVar[str]
    serializer: ClassVar[type[JsonSerializer]] = JsonSerializer
    media_type: Literal["application/vnd.dataframe"] = "application/vnd.dataframe"

    def object_to_upload(self) -> dict | None:
        try:
            function = cast(
                Callable,
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return None

        result = (
            function()
            if self.data_source is None
            else function(data_source=self.data_source)
        )

        if not isinstance(result, DataFrame):
            result = result.frame()

        return result.fillna("NaN").to_dict(orient="tight")


class Permutation(FeatureImportance, ABC):  # noqa: D101
    report: EstimatorReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str] = "feature_importance.permutation"
    name: Literal["permutation"] = "permutation"


class PermutationTrain(Permutation):  # noqa: D101
    data_source: Literal["train"] = "train"


class PermutationTest(Permutation):  # noqa: D101
    data_source: Literal["test"] = "test"


class MeanDecreaseImpurity(FeatureImportance):  # noqa: D101
    report: EstimatorReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str] = "feature_importance.mean_decrease_impurity"
    name: Literal["mean_decrease_impurity"] = "mean_decrease_impurity"


class Coefficients(FeatureImportance):  # noqa: D101
    accessor: ClassVar[str] = "feature_importance.coefficients"
    name: Literal["coefficients"] = "coefficients"
