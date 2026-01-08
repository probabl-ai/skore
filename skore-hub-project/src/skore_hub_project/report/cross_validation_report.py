"""Class definition of the payload used to send a cross-validation report to ``hub``."""

from collections import Counter, defaultdict
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from pydantic import computed_field
from scipy.stats import gaussian_kde
from sklearn.model_selection._split import _CVIterableWrapper

from skore_hub_project.artifact.media import (
    EstimatorHtmlRepr,
    PrecisionRecallTest,
    PrecisionRecallTrain,
    PredictionErrorTest,
    PredictionErrorTrain,
    RocTest,
    RocTrain,
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
        EstimatorHtmlRepr,
        PrecisionRecallTest,
        PrecisionRecallTrain,
        PredictionErrorTest,
        PredictionErrorTrain,
        RocTest,
        RocTrain,
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
        """
        Splitting strategy used to split the dataset into train and test sets.

        This includes the number of splits, the number of repeats, the seed,
        and the distribution of the train and test sets.

        The distribution of each split is computed by dividing the split into a maximum
        of 200 buckets, and averaging the number of samples belonging to the test-set in
        each of these buckets. @TODO: find a better representation of the distribution.
        """
        splits = []

        for train_indices, test_indices in self.report.split_indices:
            linspace = np.linspace(self.report.y.min(), self.report.y.max(), num=100)
            train_y = self.report.y[train_indices]
            test_y = self.report.y[test_indices]
            train_target_distribution: list[float] = []
            test_target_distribution: list[float] = []

            if self.__classes:
                train = {str(label): count for label, count in Counter(train_y).items()}
                test = {str(label): count for label, count in Counter(test_y).items()}

                for label in self.__classes:
                    train_target_distribution.append(train.get(label, 0))
                    test_target_distribution.append(test.get(label, 0))

            else:
                train_kernel = gaussian_kde(train_y)
                train_target_distribution = [float(x) for x in train_kernel(linspace)]
                test_kernel = gaussian_kde(test_y)
                test_target_distribution = [float(x) for x in test_kernel(linspace)]

            # remove this when a better solution is found
            # see #2212 for details
            buckets_number = min(len(self.report.X), 200)
            split = np.zeros(len(self.report.X), dtype=int)
            split[test_indices] = 1

            splits.append(
                {
                    "train": {
                        "sample_count": len(train_indices),
                        "target_distribution": train_target_distribution,
                        "groups": None,
                    },
                    "test": {
                        "sample_count": len(test_indices),
                        "target_distribution": test_target_distribution,
                        "groups": None,
                    },
                    "train_test_distribution": [
                        float(np.mean(bucket))
                        for bucket in np.array_split(split, buckets_number)
                    ],
                }
            )

        return {
            "strategy_name": (
                isinstance(self.report.splitter, _CVIterableWrapper)
                and "custom"
                or self.report.splitter.__class__.__name__
            ),
            "repeat_count": getattr(self.report.splitter, "n_repeats", 1),
            "seed": str(getattr(self.report.splitter, "random_state", "")),
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
