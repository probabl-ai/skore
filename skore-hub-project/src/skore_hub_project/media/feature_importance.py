from collections.abc import Callable
from functools import cached_property, reduce
from inspect import signature
from typing import ClassVar, Literal, cast

from pydantic import Field, computed_field
from skore import EstimatorReport

from .media import Media, Representation


class FeatureImportance(Media):
    report: EstimatorReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str]
    category: Literal["feature_importance"] = "feature_importance"

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def representation(self) -> Representation | None:
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

        dataframe = function(**function_kwargs)

        return Representation(
            media_type="application/vnd.dataframe",
            value=dataframe.fillna("NaN").to_dict(orient="tight"),
        )


class Permutation(FeatureImportance):
    accessor: ClassVar[str] = "feature_importance.permutation"
    key: str = "permutation"
    verbose_name: str = "Feature importance - Permutation"


class PermutationTrain(Permutation):
    attributes: dict = {"data_source": "train", "method": "permutation"}


class PermutationTest(Permutation):
    attributes: dict = {"data_source": "test", "method": "permutation"}


class MeanDecreaseImpurity(FeatureImportance):
    accessor: ClassVar[str] = "feature_importance.mean_decrease_impurity"
    key: str = "mean_decrease_impurity"
    verbose_name: str = "Feature importance - Mean Decrease Impurity (MDI)"
    attributes: dict = {"method": "mean_decrease_impurity"}


class Coefficients(FeatureImportance):
    accessor: ClassVar[str] = "feature_importance.coefficients"
    key: str = "coefficients"
    verbose_name: str = "Feature importance - Coefficients"
    attributes: dict = {"method": "coefficients"}
