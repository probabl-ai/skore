"""Class definition of the payload used to send a cross-validation report to ``hub``."""

from collections import defaultdict
from functools import cached_property
from typing import ClassVar, Literal, cast

from pydantic import Field, computed_field
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import _CVIterableWrapper
from skore import CrossValidationReport

from skore_hub_project.artefact import CrossValidationReportArtefact
from skore_hub_project.media import (
    EstimatorHtmlRepr,
)
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
    upload : bool, optional
        Upload the report to the artefacts storage, default True.
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
    MEDIAS: ClassVar[tuple[Media, ...]] = cast(tuple[Media, ...], (EstimatorHtmlRepr,))

    report: CrossValidationReport = Field(repr=False, exclude=True)

    def model_post_init(self, context):  # noqa: D102
        if "classification" in self.ml_task:
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

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def splitting_strategy_name(self) -> str:
        """The name of the splitting strategy used by the report."""
        is_sklearn_splitter = isinstance(self.report.splitter, BaseCrossValidator)
        is_iterable_splitter = isinstance(self.report.splitter, _CVIterableWrapper)
        is_standard_strategy = is_sklearn_splitter and (not is_iterable_splitter)

        return (
            is_standard_strategy and self.report.splitter.__class__.__name__ or "custom"
        )

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def splits(self) -> list[list[Literal[0, 1]]]:
        """
        The dataset splits used by the report.

        Notes
        -----
        For each split and for each sample in the dataset:
        - 0 if the sample is in the train-set,
        - 1 if the sample is in the test-set.
        """
        splits = [
            [0] * len(self.report.X) for i in range(len(self.report.split_indices))
        ]

        for i, (_, test_indices) in enumerate(self.report.split_indices):
            for test_indice in test_indices:
                splits[i][test_indice] = 1

        return cast(list[list[Literal[0, 1]]], splits)

    groups: list[int] | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def class_names(self) -> list[str] | None:
        """In classification, the class names of the dataset used in the report."""
        return self.__classes

    @computed_field  # type: ignore[prop-decorator]
    @property
    def classes(self) -> list[int] | None:
        """In classification, the class indice of each sample used in the report."""
        return self.__sample_to_class_indice

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def estimators(self) -> list[EstimatorReportPayload]:
        """The estimators used in each split by the report in a payload format."""
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

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def parameters(self) -> CrossValidationReportArtefact | dict[()]:
        """
        The checksum of the instance.

        The checksum of the instance that was assigned after being uploaded to the
        artefact storage. It is based on its ``joblib`` serialization and mainly used to
        retrieve it from the artefacts storage.

        .. deprecated
          The ``parameters`` property will be removed in favor of a new ``checksum``
          property in a near future.
        """
        if self.upload:
            return CrossValidationReportArtefact(
                project=self.project,
                report=self.report,
            )
        return {}
