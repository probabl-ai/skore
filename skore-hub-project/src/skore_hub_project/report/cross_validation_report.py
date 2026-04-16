"""Class definition of the payload used to send a cross-validation report to ``hub``."""

from collections import Counter, defaultdict
from functools import cached_property
from inspect import signature
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import computed_field
from scipy.stats import gaussian_kde
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
TARGET_DISTRIBUTION_REPR_SAMPLE_COUNT = 100
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
        from skore._externals._sklearn_compat import (  # type: ignore[attr-defined]
            _safe_indexing,
        )

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

        rng = np.random.default_rng(0)
        rng_size = min(len(target), SPLITTING_STRATEGY_REPR_SAMPLE_COUNT)

        if len(target) < SPLITTING_STRATEGY_REPR_SAMPLE_COUNT:
            target_repr = target
        elif is_classifier:
            # create an undersampled target to create a simplify representation
            if not isinstance(target, pd.Series):
                target = pd.Series(target)
            probs = target.value_counts(normalize=True)
            target_repr = rng.choice(
                probs.index.to_numpy(),  # classes
                size=rng_size,
                p=probs.to_numpy(),  # probabilities
                replace=True,
            )
            target_repr.sort()
        else:  # regression
            # uniformly sample the target because it will have no impact on the
            # representation
            target_repr = rng.choice(target, size=rng_size, replace=False)

        # create a simplified splitter without shuffling and repetitions
        simplified_cls = SPLITTERS.get(splitter.__class__.__name__, splitter.__class__)
        simplified_cls_parameters = {}

        for key in signature(simplified_cls.__init__).parameters:
            if key in splitter_metadata:
                simplified_cls_parameters[key] = splitter_metadata[key]
            elif hasattr(splitter, key):
                simplified_cls_parameters[key] = getattr(splitter, key)

        if "shuffle" in simplified_cls_parameters:
            simplified_cls_parameters["shuffle"] = False
            simplified_cls_parameters["random_state"] = None

        simplified_splitter = simplified_cls(**simplified_cls_parameters)

        # Per split: one list of length N_SAMPLES_REPR, ordered by sample index,
        # with 0 = train fold and 1 = test fold for that split.
        splits: list[list[int]] = []
        X = rng.normal(size=(rng_size, 1))

        for train_idx, test_idx in simplified_splitter.split(X, target_repr):
            split_flags = np.full(rng_size, -1, dtype=int)
            split_flags[train_idx] = 0
            split_flags[test_idx] = 1

            splits.append(split_flags.tolist())

        # compute target distributions
        train_target_distributions = []
        train_target_distributions_sample_count = []
        test_target_distributions = []
        test_target_distributions_sample_count = []
        for train_indices, test_indices in self.report.split_indices:
            train_y = _safe_indexing(self.report.y, train_indices)
            test_y = _safe_indexing(self.report.y, test_indices)
            train_target_distribution: list[float] = []
            test_target_distribution: list[float] = []

            if self.__classes:
                train = {str(label): count for label, count in Counter(train_y).items()}
                test = {str(label): count for label, count in Counter(test_y).items()}

                for label in self.__classes:
                    train_target_distribution.append(train.get(label, 0))
                    test_target_distribution.append(test.get(label, 0))

            else:
                linspace = np.linspace(
                    float(train_y.min()),
                    float(train_y.max()),
                    num=TARGET_DISTRIBUTION_REPR_SAMPLE_COUNT,
                )
                train_kernel = gaussian_kde(train_y)
                train_target_distribution = [float(x) for x in train_kernel(linspace)]
                test_kernel = gaussian_kde(test_y)
                test_target_distribution = [float(x) for x in test_kernel(linspace)]

            train_target_distributions.append(train_target_distribution)
            train_target_distributions_sample_count.append(len(train_indices))
            test_target_distributions.append(test_target_distribution)
            test_target_distributions_sample_count.append(len(test_indices))

        return {
            "splitter": splitter_metadata,
            "splits": splits,
            "train_target_distributions": train_target_distributions,
            "train_target_distributions_sample_counts": (
                train_target_distributions_sample_count
            ),
            "test_target_distributions": test_target_distributions,
            "test_target_distributions_sample_counts": (
                test_target_distributions_sample_count
            ),
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
