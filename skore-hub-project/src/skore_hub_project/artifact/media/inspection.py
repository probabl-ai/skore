"""Definition of the payload used to associate inspection media with report."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from functools import reduce
from typing import ClassVar, Literal, cast

from orjson import OPT_NON_STR_KEYS, OPT_SERIALIZE_NUMPY, dumps

from skore_hub_project.artifact.media.media import Media, Report
from skore_hub_project.protocol import Display, EstimatorReport


class Inspection(Media[Report], ABC):  # noqa: D101
    accessor: ClassVar[str]
    content_type: Literal["application/vnd.dataframe"] = "application/vnd.dataframe"

    def content_to_upload(self) -> bytes | None:  # noqa: D102
        try:
            function = cast(
                "Callable[..., Display]",
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return None

        display = function()
        frame = display.frame()

        return dumps(
            frame.fillna("NaN").to_dict(orient="tight"),
            option=(OPT_NON_STR_KEYS | OPT_SERIALIZE_NUMPY),
        )


class PermutationImportance(Inspection[EstimatorReport], ABC):  # noqa: D101
    accessor: ClassVar[str] = "inspection.permutation_importance"
    name: Literal["permutation_importance"] = "permutation_importance"

    def content_to_upload(self) -> bytes | None:  # noqa: D102
        display = self.report.inspection.get_cached_permutation_importance(
            data_source=self.data_source,
            at_step=0,
            metric=None,
        )
        if display is None:
            return None

        frame = display.frame()
        return dumps(
            frame.fillna("NaN").to_dict(orient="tight"),
            option=(OPT_NON_STR_KEYS | OPT_SERIALIZE_NUMPY),
        )


class PermutationImportanceTrain(PermutationImportance):  # noqa: D101
    data_source: Literal["train"] = "train"


class PermutationImportanceTest(PermutationImportance):  # noqa: D101
    data_source: Literal["test"] = "test"


class ImpurityDecrease(Inspection[EstimatorReport]):  # noqa: D101
    accessor: ClassVar[str] = "inspection.impurity_decrease"
    name: Literal["impurity_decrease"] = "impurity_decrease"
    data_source: None = None


class Coefficients(Inspection[Report]):  # noqa: D101
    accessor: ClassVar[str] = "inspection.coefficients"
    name: Literal["coefficients"] = "coefficients"
    data_source: None = None
