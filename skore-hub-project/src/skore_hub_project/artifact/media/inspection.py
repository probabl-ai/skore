"""Definition of the payload used to associate inspection media with report."""

from __future__ import annotations

import inspect
from abc import ABC
from collections.abc import Callable
from functools import reduce
from typing import Any, ClassVar, Literal, cast

from skore_hub_project.artifact.media.media import Media, Report
from skore_hub_project.json import dumps
from skore_hub_project.protocol import Display


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
        # FIXME: in the future, all inspection methods should have an aggregate
        # parameter and we should be sending unaggregated data to the hub.
        if "aggregate" in inspect.signature(display.frame).parameters:
            frame = display.frame(aggregate=None)
        else:
            frame = display.frame()

        return dumps(
            frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
        )


class PermutationImportance(Inspection[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "inspection.permutation_importance"
    name: Literal["permutation_importance"] = "permutation_importance"

    def _get_display(self) -> Any:
        if hasattr(self.report.inspection, "_get_cached_permutation_importances"):
            kwargs_list = list(
                self.report.inspection._get_cached_permutation_importances(
                    self.data_source
                )
            )
            for kwargs in reversed(kwargs_list):
                if kwargs["at_step"] == 0 and kwargs.get("metric") is None:
                    return self.report.inspection.permutation_importances(**kwargs)

        # old skore (<= 0.14)
        cache = getattr(self.report, "_cache", {})
        for key, display in reversed(list(cache.items())):
            if len(key) < 6:
                continue

            parent_hash, name, data_source, at_step, _, metric, *_ = key

            if (
                parent_hash == self.report._hash
                and name == "permutation_importance"
                and data_source == self.data_source
                and at_step == 0
                and metric is None
            ):
                return display

        return None

    def content_to_upload(self) -> bytes | None:  # noqa: D102
        display = self._get_display()
        if display is None:
            return None

        frame = display.frame(aggregate=None)
        return dumps(
            frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
        )


class PermutationImportanceTrain(PermutationImportance[Report]):  # noqa: D101
    data_source: Literal["train"] = "train"


class PermutationImportanceTest(PermutationImportance[Report]):  # noqa: D101
    data_source: Literal["test"] = "test"


class ImpurityDecrease(Inspection[Report]):  # noqa: D101
    accessor: ClassVar[str] = "inspection.impurity_decrease"
    name: Literal["impurity_decrease"] = "impurity_decrease"
    data_source: None = None


class Coefficients(Inspection[Report]):  # noqa: D101
    accessor: ClassVar[str] = "inspection.coefficients"
    name: Literal["coefficients"] = "coefficients"
    data_source: None = None
