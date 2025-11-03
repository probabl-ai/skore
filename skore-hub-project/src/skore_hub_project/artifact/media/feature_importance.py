"""Definition of the payload used to associate feature importance media with report."""

from abc import ABC
from collections.abc import Callable
from functools import reduce
from typing import ClassVar, Literal, cast

from orjson import dumps, OPT_NON_STR_KEYS, OPT_SERIALIZE_NUMPY
from pandas import DataFrame

from skore_hub_project.artifact.media.media import Media, Report
from skore_hub_project.protocol import EstimatorReport, Display


class FeatureImportance(Media[Report], ABC):  # noqa: D101
    accessor: ClassVar[str]
    content_type: Literal["application/vnd.dataframe"] = "application/vnd.dataframe"

    def content_to_upload(self) -> bytes | None:  # noqa: D102
        try:
            function = cast(
                Callable[..., Display | DataFrame],
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return None

        result = function()

        if isinstance(result, Display):
            result = result.frame()

        return dumps(
            result.fillna("NaN").to_dict(orient="tight"),
            option=(OPT_NON_STR_KEYS | OPT_SERIALIZE_NUMPY),
        )


class Permutation(FeatureImportance[EstimatorReport], ABC):  # noqa: D101
    accessor: ClassVar[str] = "feature_importance.permutation"
    name: Literal["permutation"] = "permutation"

    def content_to_upload(self) -> bytes | None:  # noqa: D102
        for key, obj in reversed(self.report._cache.items()):
            if len(key) < 7:
                continue

            if len(key) == 7:
                parent_hash, metric, data_source, scoring, *_ = key
            else:
                parent_hash, metric, data_source, data_source_hash, scoring, *_ = key

            if (
                parent_hash == self.report._hash
                and metric == "permutation_importance"
                and data_source == self.data_source
                and scoring is None
            ):
                return dumps(
                    obj.fillna("NaN").to_dict(orient="tight"),
                    option=(OPT_NON_STR_KEYS | OPT_SERIALIZE_NUMPY),
                )

        return None


class PermutationTrain(Permutation):  # noqa: D101
    data_source: Literal["train"] = "train"


class PermutationTest(Permutation):  # noqa: D101
    data_source: Literal["test"] = "test"


class MeanDecreaseImpurity(FeatureImportance[EstimatorReport]):  # noqa: D101
    accessor: ClassVar[str] = "feature_importance.mean_decrease_impurity"
    name: Literal["mean_decrease_impurity"] = "mean_decrease_impurity"
    data_source: None = None


class Coefficients(FeatureImportance[Report]):  # noqa: D101
    accessor: ClassVar[str] = "feature_importance.coefficients"
    name: Literal["coefficients"] = "coefficients"
    data_source: None = None
