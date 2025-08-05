from collections import defaultdict
from functools import cached_property
from typing import Any, TYPE_CHECKING, Literal

from pydantic import Field, computed_field
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import _CVIterableWrapper

from .report import ReportPayload
from .estimator_report import EstimatorReportPayload

Metric = Any
CrossValidationReport = Any


class CrossValidationReportPayload(ReportPayload):
    report: CrossValidationReport = Field(repr=False, exclude=True)

    def model_post_init(self, context):
        self.report.ml_task = self.report._ml_task  # to revert after rebase main

        if "classification" in self.report.ml_task:
            class_to_class_indice = defaultdict(lambda: len(class_to_class_indice))

            self.__sample_to_class_indice = [
                class_to_class_indice[sample] for sample in self.report.y
            ]

            assert len(self.__sample_to_class_indice) == len(self.report.X)

            self.__classes = [str(class_) for class_ in class_to_class_indice]

            assert max(self.__sample_to_class_indice) == (len(self.__classes) - 1)
        else:
            self.__sample_to_class_indice = None
            self.__classes = None

    @computed_field
    @cached_property
    def splitting_strategy_name(self) -> str:
        is_sklearn_splitter = isinstance(self.report.splitter, BaseCrossValidator)
        is_iterable_splitter = isinstance(self.report.splitter, _CVIterableWrapper)
        is_standard_strategy = is_sklearn_splitter and (not is_iterable_splitter)

        # fmt: off
        return (
            (is_standard_strategy and self.report.splitter.__class__.__name__)
            or "custom"
        )
        # fmt: on

    @computed_field
    @cached_property
    def splits(self) -> list[list[Literal[0, 1]]]:
        """
        Notes
        -----
        For each split and for each sample in the dataset:
        - 0 if the sample is in the train-set,
        - 1 if the sample is in the test-set.
        """
        splits = [[0] * len(self.report.X)] * len(self.report.split_indices)

        for i, (_, test_indices) in enumerate(self.report.split_indices):
            for test_indice in test_indices:
                splits[i][test_indice] = 1

        return splits

    groups: list[int] | None = None

    @computed_field
    @property
    def class_names(self) -> list[str] | None:
        return self.__classes

    @computed_field
    @property
    def classes(self) -> list[int] | None:
        return self.__sample_to_class_indice

    @computed_field
    @cached_property
    def estimators(self) -> list[EstimatorReportPayload]:
        return [
            EstimatorReportPayload(
                project=self.project,
                report=report,
                upload=False,
                key=f"{self.key}:estimator-report",
                run_id=self.run_id,
            )
            for report in self.report.estimator_reports_
        ]

    @computed_field
    @cached_property
    def metrics(self) -> list[Metric] | None:
        return None

    @computed_field
    @cached_property
    def medias(self) -> list[Metric] | None:
        return None
