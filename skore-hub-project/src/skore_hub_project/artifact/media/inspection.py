"""Definition of the payload used to associate inspection media with report."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from functools import reduce
from typing import ClassVar, Literal, cast

from pandas import concat

from skore_hub_project.artifact.media.media import Media, Report
from skore_hub_project.json import dumps
from skore_hub_project.protocol import CrossValidationReport, Display, EstimatorReport


class Inspection(Media[Report], ABC):  # noqa: D101
    accessor: ClassVar[str]
    content_type: Literal["application/vnd.dataframe"] = "application/vnd.dataframe"

    def compute(self) -> None:  # noqa: D102
        if self.computed:
            return

        try:
            function = cast(
                "Callable[..., Display]",
                reduce(getattr, self.accessor.split("."), self.report),
            )
        except AttributeError:
            return

        display = function()
        frame = display.frame(aggregate=None)
        frame_as_dict = (
            frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
        )

        self.filepath.write_bytes(dumps(frame_as_dict))
        self.computed = True


class PermutationImportance(Inspection[Report], ABC):  # noqa: D101
    accessor: ClassVar[str] = "inspection.permutation_importance"
    name: Literal["permutation_importance"] = "permutation_importance"

    @staticmethod
    def __display_from_cross_validation_report(
        data_source: str | None,
        report: CrossValidationReport,
    ) -> Display | None:
        from skore import PermutationImportanceDisplay

        frames = []

        for i, display in enumerate(
            PermutationImportance.__display_from_estimator_report(
                data_source, report
            )
            for report in report.estimator_reports_
        ):
            if display is None:
                return None

            frames.append(display.importances.assign(split=i))  # type: ignore[attr-defined]

        return cast(
            Display,
            PermutationImportanceDisplay(
                importances=concat(frames, ignore_index=True),
                report_type="cross-validation",
            ),
        )

    @staticmethod
    def __display_from_estimator_report(
        data_source: str | None,
        report: EstimatorReport,
    ) -> Display | None:
        for key, display in reversed(list(report._cache.items())):
            if len(key) != 3:
                continue

            key_data_source, key_name, key_kwargs = key
            key_kwargs = str(key_kwargs)

            if (
                key_data_source == data_source
                and key_name == "permutation_importance"
                and "('at_step', 0)" in key_kwargs
                and "('metric', None)" in key_kwargs
            ):
                return cast(Display, display)

        return None

    def compute(self) -> None:  # noqa: D102
        if self.computed:
            return

        display = (
            self.__display_from_estimator_report(self.data_source, self.report)
            if isinstance(self.report, EstimatorReport)
            else self.__display_from_cross_validation_report(
                self.data_source, self.report
            )
        )

        if display is None:
            return

        frame = display.frame(aggregate=None)
        frame_as_dict = (
            frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
        )

        self.filepath.write_bytes(dumps(frame_as_dict))
        self.computed = True


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
