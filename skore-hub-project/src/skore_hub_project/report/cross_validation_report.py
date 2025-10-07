"""Class definition of the payload used to send a cross-validation report to ``hub``."""

from collections import defaultdict
from functools import cached_property
from typing import ClassVar, cast

import numpy as np
from pydantic import Field, computed_field
from sklearn.model_selection._split import _CVIterableWrapper

from skore_hub_project.artifact import CrossValidationReportArtifact
from skore_hub_project.media import EstimatorHtmlRepr
from skore_hub_project.media.data import TableReport
from skore_hub_project.media.media import Media
from skore_hub_project.metric import (
    AccuracyTestMean,
    AccuracyTestStd,
    AccuracyTrainMean,
    AccuracyTrainStd,
    BrierScoreTestMean,
    BrierScoreTestStd,
    BrierScoreTrainMean,
    BrierScoreTrainStd,
    FitTimeMean,
    FitTimeStd,
    LogLossTestMean,
    LogLossTestStd,
    LogLossTrainMean,
    LogLossTrainStd,
    PrecisionTestMean,
    PrecisionTestStd,
    PrecisionTrainMean,
    PrecisionTrainStd,
    PredictTimeTestMean,
    PredictTimeTestStd,
    PredictTimeTrainMean,
    PredictTimeTrainStd,
    R2TestMean,
    R2TestStd,
    R2TrainMean,
    R2TrainStd,
    RecallTestMean,
    RecallTestStd,
    RecallTrainMean,
    RecallTrainStd,
    RmseTestMean,
    RmseTestStd,
    RmseTrainMean,
    RmseTrainStd,
    RocAucTestMean,
    RocAucTestStd,
    RocAucTrainMean,
    RocAucTrainStd,
)
from skore_hub_project.metric.metric import Metric
from skore_hub_project.protocol import CrossValidationReport
from skore_hub_project.report.estimator_report import EstimatorReportPayload
from skore_hub_project.report.report import ReportPayload


class CrossValidationReportPayload(ReportPayload):
    """
    Payload used to send a cross-validation report to ``hub``.

    Attributes
    ----------
    METRICS : ClassVar[tuple[Metric, ...]]
        The metric classes that have to be computed from the report.
    MEDIAS : ClassVar[tuple[Media, ...]]
        The media classes that have to be computed from the report.
    project : Project
        The project to which the report payload should be sent.
    report : CrossValidationReport
        The report on which to calculate the payload to be sent.
    key : str
        The key to associate to the report.
    """

    METRICS: ClassVar[tuple[Metric, ...]] = cast(
        tuple[Metric, ...],
        (
            AccuracyTestMean,
            AccuracyTestStd,
            AccuracyTrainMean,
            AccuracyTrainStd,
            BrierScoreTestMean,
            BrierScoreTestStd,
            BrierScoreTrainMean,
            BrierScoreTrainStd,
            LogLossTestMean,
            LogLossTestStd,
            LogLossTrainMean,
            LogLossTrainStd,
            PrecisionTestMean,
            PrecisionTestStd,
            PrecisionTrainMean,
            PrecisionTrainStd,
            R2TestMean,
            R2TestStd,
            R2TrainMean,
            R2TrainStd,
            RecallTestMean,
            RecallTestStd,
            RecallTrainMean,
            RecallTrainStd,
            RmseTestMean,
            RmseTestStd,
            RmseTrainMean,
            RmseTrainStd,
            RocAucTestMean,
            RocAucTestStd,
            RocAucTrainMean,
            RocAucTrainStd,
            # timings must be calculated last
            FitTimeMean,
            FitTimeStd,
            PredictTimeTestMean,
            PredictTimeTestStd,
            PredictTimeTrainMean,
            PredictTimeTrainStd,
        ),
    )
    MEDIAS: ClassVar[tuple[Media, ...]] = cast(
        tuple[Media, ...],
        (
            EstimatorHtmlRepr,
            TableReport,
        ),
    )

    report: CrossValidationReport = Field(repr=False, exclude=True)

    def model_post_init(self, context):  # noqa: D102
        if "classification" in self.ml_task:
            class_to_class_indice = defaultdict(lambda: len(class_to_class_indice))

            self.__sample_to_class_index = [
                class_to_class_indice[sample] for sample in self.report.y
            ]

            assert len(self.__sample_to_class_index) == len(self.report.X)

            self.__classes = [str(class_) for class_ in class_to_class_indice]

            assert max(self.__sample_to_class_index) == (len(self.__classes) - 1)
        else:
            self.__sample_to_class_index = None
            self.__classes = None

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def dataset_size(self) -> int:
        """Size of the dataset."""
        return len(self.report.X)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def splitting_strategy_name(self) -> str:
        """The name of the splitting strategy used by the report."""
        is_iterable_splitter = isinstance(self.report.splitter, _CVIterableWrapper)

        return (
            is_iterable_splitter and "custom" or self.report.splitter.__class__.__name__
        )

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def splits(self) -> list[list[float]]:
        """
        Distribution between train and test by split.

        The distribution of each split is computed by dividing the split into a maximum
        of 200 buckets, and averaging the number of samples belonging to the test-set in
        each of these buckets.
        """
        distributions = []
        buckets_number = min(len(self.report.X), 200)

        for _, test_indices in self.report.split_indices:
            split = np.zeros(len(self.report.X), dtype=int)
            split[test_indices] = 1

            distributions.append(
                [
                    float(np.mean(bucket))
                    for bucket in np.array_split(split, buckets_number)
                ]
            )

        return distributions

    groups: list[int] | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def class_names(self) -> list[str] | None:
        """In classification, the class names of the dataset used in the report."""
        return self.__classes

    @computed_field  # type: ignore[prop-decorator]
    @property
    def classes(self) -> list[int] | None:
        """
        In classification, the distribution of the classes in the dataset.

        The distribution is computed by dividing the dataset into a maximum of 200
        buckets, and noting the dominant class in each of these buckets.
        """
        if self.__sample_to_class_index is None:
            return None

        buckets_number = min(len(self.__sample_to_class_index), 200)
        buckets = np.array_split(self.__sample_to_class_index, buckets_number)

        return [int(np.bincount(bucket).argmax()) for bucket in buckets]

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def estimators(self) -> list[EstimatorReportPayload]:
        """The estimators used in each split by the report in a payload format."""
        return [
            EstimatorReportPayload(
                project=self.project,
                report=report,
                key=f"{self.key}:estimator-report",
            )
            for report in self.report.estimator_reports_
        ]

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def parameters(self) -> CrossValidationReportArtifact | dict[()]:
        """
        The checksum of the instance.

        The checksum of the instance that was assigned before being uploaded to the
        artifact storage. It is based on its ``joblib`` serialization and mainly used to
        retrieve it from the artifacts storage.

        .. deprecated
          The ``parameters`` property will be removed in favor of a new ``checksum``
          property in a near future.
        """
        return CrossValidationReportArtifact(project=self.project, report=self.report)
