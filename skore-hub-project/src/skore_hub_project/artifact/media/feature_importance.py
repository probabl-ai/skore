"""Definition of the payload used to associate feature importance media with report."""

from abc import ABC
from collections.abc import Callable
from functools import reduce
from typing import ClassVar, Literal, cast

from skore_hub_project.artifact.media.media import Media, Report
from skore_hub_project.protocol import EstimatorReport


class FeatureImportance(Media[Report], ABC):  # noqa: D101
    accessor: ClassVar[str]
    content_type: Literal["application/vnd.dataframe"] = "application/vnd.dataframe"

    def content_to_upload(self) -> bytes | None:  # noqa: D102
        import orjson

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

        if hasattr(result, "frame"):
            result = result.frame()

        return orjson.dumps(
            result.fillna("NaN").to_dict(orient="tight"),
            option=(orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY),
        )


class Permutation(FeatureImportance[EstimatorReport], ABC):  # noqa: D101
    accessor: ClassVar[str] = "feature_importance.permutation"
    name: Literal["permutation"] = "permutation"


class PermutationTrain(Permutation):  # noqa: D101
    data_source: Literal["train"] = "train"


class PermutationTest(Permutation):  # noqa: D101
    data_source: Literal["test"] = "test"


class MeanDecreaseImpurity(FeatureImportance[EstimatorReport]):  # noqa: D101
    accessor: ClassVar[str] = "feature_importance.mean_decrease_impurity"
    name: Literal["mean_decrease_impurity"] = "mean_decrease_impurity"
    data_source: None = None


class Coefficients(FeatureImportance):  # noqa: D101
    accessor: ClassVar[str] = "feature_importance.coefficients"
    name: Literal["coefficients"] = "coefficients"
    data_source: None = None
