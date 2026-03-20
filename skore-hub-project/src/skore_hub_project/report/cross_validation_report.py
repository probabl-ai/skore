"""Class definition of the payload used to send a cross-validation report to ``hub``."""

from collections import defaultdict
from functools import cached_property
from inspect import signature
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import computed_field
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)

from skore_hub_project.artifact.media import (
    Coefficients,
    ConfusionMatrixDataFrameTest,
    ConfusionMatrixDataFrameTrain,
    ConfusionMatrixSVGTest,
    ConfusionMatrixSVGTrain,
    EstimatorHtmlRepr,
    ImpurityDecrease,
    PermutationImportanceTest,
    PermutationImportanceTrain,
    PrecisionRecallDataFrameTest,
    PrecisionRecallDataFrameTrain,
    PrecisionRecallSVGTest,
    PrecisionRecallSVGTrain,
    PredictionErrorDataFrameTest,
    PredictionErrorDataFrameTrain,
    PredictionErrorSVGTest,
    PredictionErrorSVGTrain,
    RocDataFrameTest,
    RocDataFrameTrain,
    RocSVGTest,
    RocSVGTrain,
)
from skore_hub_project.artifact.media.data import TableReport
from skore_hub_project.artifact.media.media import Media
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

SPLITTING_STRATEGY_REPR_SAMPLE_COUNT = 100
SPLITTERS = {
    "KFold": KFold,
    "RepeatedKFold": KFold,
    "StratifiedKFold": StratifiedKFold,
    "RepeatedStratifiedKFold": StratifiedKFold,
    "ShuffleSplit": ShuffleSplit,
    "StratifiedShuffleSplit": StratifiedShuffleSplit,
    "TimeSeriesSplit": TimeSeriesSplit,
}


class CrossValidationReportPayload(ReportPayload[CrossValidationReport]):
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

    METRICS: ClassVar[tuple[type[Metric[CrossValidationReport]], ...]] = (
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
        # timings must be calculated last, or predictions must be cached before
        FitTimeMean,
        FitTimeStd,
        PredictTimeTestMean,
        PredictTimeTestStd,
        PredictTimeTrainMean,
        PredictTimeTrainStd,
    )
    MEDIAS: ClassVar[tuple[type[Media[CrossValidationReport]], ...]] = (
        Coefficients,
        ConfusionMatrixDataFrameTest,
        ConfusionMatrixDataFrameTrain,
        ConfusionMatrixSVGTest,
        ConfusionMatrixSVGTrain,
        EstimatorHtmlRepr,
        ImpurityDecrease,
        PermutationImportanceTest,
        PermutationImportanceTrain,
        PrecisionRecallDataFrameTest,
        PrecisionRecallDataFrameTrain,
        PrecisionRecallSVGTest,
        PrecisionRecallSVGTrain,
        PredictionErrorDataFrameTest,
        PredictionErrorDataFrameTrain,
        PredictionErrorSVGTest,
        PredictionErrorSVGTrain,
        RocDataFrameTest,
        RocDataFrameTrain,
        RocSVGTest,
        RocSVGTrain,
        TableReport,
    )

    def model_post_init(self, _: Any) -> None:  # noqa: D102
        self.__sample_to_class_index: list[int] | None
        self.__classes: list[str] | None

        if "classification" in self.ml_task:
            class_to_class_indice: defaultdict[Any, int] = defaultdict(
                lambda: len(class_to_class_indice)
            )

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
    def splitting_strategy(self) -> dict[str, Any]:
        """The splitting strategy used in the report."""
        splitter = self.report.splitter
        target = self.report.y
        is_classifier = "classification" in self.ml_task

        n_repeats = getattr(splitter, "n_repeats", None)
        n_splits = splitter.get_n_splits() // (n_repeats or 1)
        splitter_metadata = {
            "type": splitter.__class__.__name__,
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            # first check if shuffle is available;
            # otherwise, we could have splitter always
            # shuffling and only exposing a random_state
            "shuffle": getattr(splitter, "shuffle", hasattr(splitter, "random_state")),
            "random_state": getattr(splitter, "random_state", None),
        }

        # create an undersampled target to create a simplify representation
        rng = np.random.default_rng(0)
        if is_classifier:
            if not isinstance(target, pd.Series):
                target = pd.Series(target)
            probs = target.value_counts(normalize=True)
            target_repr = rng.choice(
                probs.index.to_numpy(),  # classes
                size=SPLITTING_STRATEGY_REPR_SAMPLE_COUNT,
                p=probs.to_numpy(),  # probabilities
                replace=True,
            )
            target_repr.sort()
        else:  # regression
            # uniformly sample the target because it will have no impact on the
            # representation
            target_repr = rng.choice(
                target,
                size=SPLITTING_STRATEGY_REPR_SAMPLE_COUNT,
                replace=False,
            )

        # create a simplified splitter without randomization and repetitions
        simplified_cls = SPLITTERS.get(splitter.__class__.__name__, splitter.__class__)
        simplified_cls_parameters = {}

        for parameter in signature(simplified_cls.__init__).parameters:
            if parameter in splitter_metadata:
                simplified_cls_parameters[parameter] = splitter_metadata[parameter]
            elif hasattr(splitter, parameter):
                simplified_cls_parameters[parameter] = getattr(splitter, parameter)

        simplified_splitter = simplified_cls(**simplified_cls_parameters)

        # Per split: one list of length N_SAMPLES_REPR, ordered by sample index,
        # with 0 = train fold and 1 = test fold for that split.
        splits: list[list[int]] = []
        X = rng.normal(size=(SPLITTING_STRATEGY_REPR_SAMPLE_COUNT, 1))

        for train_idx, test_idx in simplified_splitter.split(X, target_repr):
            split_flags = [-1] * SPLITTING_STRATEGY_REPR_SAMPLE_COUNT
            for i in train_idx:
                split_flags[int(i)] = 0
            for i in test_idx:
                split_flags[int(i)] = 1

            splits.append(split_flags)

        return {
            "splitter": splitter_metadata,
            "dataset_size": len(self.report.X),
            "splits": splits,
        }

    groups: list[int] | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def class_names(self) -> list[str] | None:
        """In classification, the class names of the dataset used in the report."""
        return self.__classes

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def target_range(self) -> list[float] | None:
        """The range of the target values of the dataset used in the report."""
        if self.__classes:
            return None
        return [float(self.report.y.min()), float(self.report.y.max())]

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
