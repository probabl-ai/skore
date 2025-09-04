"""Class definition of the payload used to send feature importance medias to ``hub``."""

from collections.abc import Callable
from functools import cached_property, reduce
from inspect import signature
from typing import ClassVar, Literal, cast

from pandas import DataFrame
from pydantic import Field, computed_field
from skore import CrossValidationReport, EstimatorReport

from .media import Media, Representation


class FeatureImportance(Media):  # noqa: D101
    report: EstimatorReport | CrossValidationReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str]
    category: Literal["feature_importance"] = "feature_importance"

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def representation(self) -> Representation | None:  # noqa: D102
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

        result = function(**function_kwargs)

        if not isinstance(result, DataFrame):
            result = result.frame()

        serialized = result.fillna("NaN").to_dict(orient="tight")

        return Representation(media_type="application/vnd.dataframe", value=serialized)


class Permutation(FeatureImportance):  # noqa: D101
    report: EstimatorReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str] = "feature_importance.permutation"
    key: str = "permutation"
    verbose_name: str = "Feature importance - Permutation"


class PermutationTrain(Permutation):  # noqa: D101
    attributes: dict = {"data_source": "train", "method": "permutation"}


class PermutationTest(Permutation):  # noqa: D101
    attributes: dict = {"data_source": "test", "method": "permutation"}


class MeanDecreaseImpurity(FeatureImportance):  # noqa: D101
    report: EstimatorReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str] = "feature_importance.mean_decrease_impurity"
    key: str = "mean_decrease_impurity"
    verbose_name: str = "Feature importance - Mean Decrease Impurity (MDI)"
    attributes: dict = {"method": "mean_decrease_impurity"}


class Coefficients(FeatureImportance):  # noqa: D101
    accessor: ClassVar[str] = "feature_importance.coefficients"
    key: str = "coefficients"
    verbose_name: str = "Feature importance - Coefficients"
    attributes: dict = {"method": "coefficients"}
