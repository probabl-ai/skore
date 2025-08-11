from functools import cached_property, reduce
from inspect import signature
from typing import ClassVar, Literal

from pydantic import Field, computed_field
from skore import EstimatorReport

from .media import Media, Representation


class FeatureImportance(Media):
    report: EstimatorReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str]
    category: Literal["feature_importance"] = "feature_importance"

    @computed_field
    @cached_property
    def representation(self) -> Representation | None:
        try:
            function = reduce(getattr, self.accessor.split("."), self.report)
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
    accessor: ClassVar[Literal["feature_importance.permutation"]] = (
        "feature_importance.permutation"
    )
    key: Literal["permutation"] = "permutation"
    verbose_name: Literal["Feature importance - Permutation"] = (
        "Feature importance - Permutation"
    )


class PermutationTrain(Permutation):
    attributes: Literal[{"data_source": "train", "method": "permutation"}] = {
        "data_source": "train",
        "method": "permutation",
    }


class PermutationTest(Permutation):
    attributes: Literal[{"data_source": "test", "method": "permutation"}] = {
        "data_source": "test",
        "method": "permutation",
    }


class MeanDecreaseImpurity(FeatureImportance):
    accessor: ClassVar[Literal["feature_importance.mean_decrease_impurity"]] = (
        "feature_importance.mean_decrease_impurity"
    )
    key: Literal["mean_decrease_impurity"] = "mean_decrease_impurity"
    verbose_name: Literal["Feature importance - Mean Decrease Impurity (MDI)"] = (
        "Feature importance - Mean Decrease Impurity (MDI)"
    )
    attributes: Literal[{"method": "mean_decrease_impurity"}] = {
        "method": "mean_decrease_impurity"
    }


class Coefficients(FeatureImportance):
    accessor: ClassVar[Literal["feature_importance.coefficients"]] = (
        "feature_importance.coefficients"
    )
    key: Literal["coefficients"] = "coefficients"
    verbose_name: Literal["Feature importance - Coefficients"] = (
        "Feature importance - Coefficients"
    )
    attributes: Literal[{"method": "coefficients"}] = {"method": "coefficients"}
