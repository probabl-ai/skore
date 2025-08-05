from collections import defaultdict
from typing import Any

from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import _CVIterableWrapper

from .report import EstimatorReportPayload


Metric = Any
CrossValidationReport = Any


class CrossValidationReportPayload(ReportPayload):
    report: CrossValidationReport = Field(repr=False, exclude=True)

    def model_post_init(self, _):
        if self.report.ml_task.includes("classification"):
            class_to_class_indice = defaultdict(lambda: len(class_to_class_indice))

            self.__classes = list(map(str, class_to_class_indice))
            self.__sample_to_class_indice = list(map(class_to_class_indice, report.y))

            assert len(self.__sample_to_class_indice) == len(self.report.X)
            assert max(self.__sample_to_class_indice) == len(self.__classes)
        else:
            self.__classes = None
            self.__sample_to_class_indice = None

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
        splits = [[0] * len(report.X)] * len(report.split_indices)

        for i, (_, test_indices) in enumerate(report.split_indices):
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
        return self.__sample_to_class_indices

    @computed_field
    @cached_property
    def estimators(self) -> list[EstimatorReportPayload]:
        return [
            EstimatorReportPayload(report=report, upload=False)
            for report in report.estimator_reports_
        ]

    @computed_field
    @cached_property
    def metrics(self) -> list[Metric] | None:
        return None

    @computed_field
    @cached_property
    def medias(self) -> list[Metric] | None:
        return None
