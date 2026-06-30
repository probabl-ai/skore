"""Class definition of the payload used to send a cross-validation report to ``hub``."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sized
from functools import cached_property
from inspect import signature
from itertools import chain
from typing import Any, ClassVar, cast

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

from skore import CrossValidationReport
from skore._plugins.hub.artifact.media import (
    Coefficients,
    ConfusionMatrixDataFrameTestAll,
    ConfusionMatrixDataFrameTestNone,
    ConfusionMatrixDataFrameTrainAll,
    ConfusionMatrixDataFrameTrainNone,
    EstimatorHtmlRepr,
    ImpurityDecrease,
    PermutationImportanceTest,
    PermutationImportanceTrain,
    PrecisionRecallDataFrameTest,
    PrecisionRecallDataFrameTrain,
    PredictionErrorDataFrameTest,
    PredictionErrorDataFrameTrain,
    RocDataFrameTest,
    RocDataFrameTrain,
)
from skore._plugins.hub.artifact.media.data import TableReport
from skore._plugins.hub.artifact.media.media import Media
from skore._plugins.hub.metric import Metric
from skore._plugins.hub.report.estimator_report import EstimatorReportPayload
from skore._plugins.hub.report.report import ReportPayload

SPLITTING_STRATEGY_MAX_INDEX_COUNT = 10_000
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


def _regression_target_kdes(
    y: np.ndarray,
) -> tuple[list[list[float]], list[tuple[float, float]]]:
    """Estimate per-target density curves for regression splitting strategy."""
    sample_count = TARGET_DISTRIBUTION_REPR_SAMPLE_COUNT
    y = np.asarray(y)
    if y.ndim == 1:
        lo, hi = float(y.min()), float(y.max())
        linspace = np.linspace(lo, hi, num=sample_count)
        kde = gaussian_kde(y)
        return [[float(x) for x in kde(linspace)]], [(lo, hi)]

    curves: list[list[float]] = []
    ranges: list[tuple[float, float]] = []
    for col in y.T:
        col_arr = np.asarray(col).ravel()
        lo, hi = float(col_arr.min()), float(col_arr.max())
        linspace = np.linspace(lo, hi, num=sample_count)
        curves.append([float(x) for x in gaussian_kde(col_arr).evaluate(linspace)])
        ranges.append((lo, hi))
    return curves, ranges


def _flatten_regression_target_kdes(curves: list[list[float]]) -> list[float]:
    """Flatten per-target density curves for hub payload storage."""
    if len(curves) == 1:
        return curves[0]
    return [value for curve in curves for value in curve]


class CrossValidationReportPayload(ReportPayload[CrossValidationReport]):
    """
    Payload used to send a cross-validation report to ``hub``.

    Attributes
    ----------
    MEDIAS : ClassVar[tuple[Media, ...]]
        The media classes that have to be computed from the report.
    project : Project
        The project to which the report payload should be sent.
    report : CrossValidationReport
        The report on which to calculate the payload to be sent.
    key : str
        The key to associate to the report.
    """

    MEDIAS: ClassVar[tuple[type[Media[CrossValidationReport]], ...]] = (
        Coefficients,
        ConfusionMatrixDataFrameTestAll,
        ConfusionMatrixDataFrameTestNone,
        ConfusionMatrixDataFrameTrainAll,
        ConfusionMatrixDataFrameTrainNone,
        EstimatorHtmlRepr,
        ImpurityDecrease,
        PermutationImportanceTest,
        PermutationImportanceTrain,
        PrecisionRecallDataFrameTest,
        PrecisionRecallDataFrameTrain,
        PredictionErrorDataFrameTest,
        PredictionErrorDataFrameTrain,
        RocDataFrameTest,
        RocDataFrameTrain,
        TableReport,
    )

    def model_post_init(self, _: Any) -> None:  # noqa: D102
        self.__sample_to_class_index: list[int] | None
        self.__classes: list[str] | None
        self.__target_names: list[str] | None
        self.__target_ranges: list[list[float]] | None

        if "classification" in self.ml_task and (self.report.y is not None):
            class_to_class_indice: defaultdict[Any, int] = defaultdict(
                lambda: len(class_to_class_indice)
            )

            self.__sample_to_class_index = [
                class_to_class_indice[sample]
                for sample in cast(Iterable[Any], self.report.y)
            ]

            assert len(self.__sample_to_class_index) == len(cast(Sized, self.report.X))

            self.__classes = [str(class_) for class_ in class_to_class_indice]

            assert max(self.__sample_to_class_index) == (len(self.__classes) - 1)
            self.__target_names = None
            self.__target_ranges = None
        elif self.ml_task == "multioutput-regression" and (self.report.y is not None):
            from skore._utils._dataframe import _normalize_y_as_dataframe

            y_df = _normalize_y_as_dataframe(self.report.y)
            self.__sample_to_class_index = None
            self.__classes = None
            self.__target_names = [str(name) for name in y_df.columns]
            y_arr = np.asarray(self.report.y)
            self.__target_ranges = [
                [float(col.min()), float(col.max())] for col in y_arr.T
            ]
        else:
            self.__sample_to_class_index = None
            self.__classes = None
            self.__target_names = None
            self.__target_ranges = None

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def dataset_size(self) -> int:
        """Size of the dataset."""
        return len(cast(Sized, self.report.X))

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def splitting_strategy(self) -> dict[str, Any]:
        """The splitting strategy used in the report."""
        from skore._externals._sklearn_compat import _safe_indexing

        if self.report.y is None:
            return {}

        splits: list[list[int] | None] = []
        splitter = self.report.splitter
        target = cast(Sized, self.report.y)
        is_classifier = "classification" in self.ml_task

        n_repeats = getattr(splitter, "n_repeats", None)
        n_splits = len(self.report.split_indices) // (n_repeats or 1)

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

        # create a simplified splitter without shuffling and repetitions when possible
        if simplified_cls := SPLITTERS.get(splitter.__class__.__name__):
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
                target_repr = rng.choice(
                    cast(np.ndarray, target), size=rng_size, replace=False
                )

            simplified_cls_parameters = {}

            for key in signature(simplified_cls.__init__).parameters:
                if key in splitter_metadata:
                    simplified_cls_parameters[key] = splitter_metadata[key]
                elif hasattr(splitter, key):
                    simplified_cls_parameters[key] = getattr(splitter, key)

            if "shuffle" in simplified_cls_parameters:
                simplified_cls_parameters["shuffle"] = False
                simplified_cls_parameters["random_state"] = None

            target = target_repr
            simplified_splitter = simplified_cls(**simplified_cls_parameters)
            split_generator = simplified_splitter.split(
                rng.normal(size=(rng_size, 1)),
                target_repr,
            )
        else:
            total_index_count = sum(
                len(cast(Sized, indices))
                for indices in chain.from_iterable(self.report.split_indices)
            )

            if total_index_count > SPLITTING_STRATEGY_MAX_INDEX_COUNT:
                splits = [None] * len(self.report.split_indices)
                split_generator = []
            else:
                split_generator = self.report.split_indices

        # Per split: one list of length N_SAMPLES_REPR, ordered by sample index,
        # with 0 = train fold and 1 = test fold for that split.
        for train_idx, test_idx in split_generator:
            split_flags = np.full(len(target), -1, dtype=int)
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
                train_curves, _ = _regression_target_kdes(train_y)
                test_curves, _ = _regression_target_kdes(test_y)
                train_target_distribution = _flatten_regression_target_kdes(
                    train_curves
                )
                test_target_distribution = _flatten_regression_target_kdes(test_curves)

            train_target_distributions.append(train_target_distribution)
            train_target_distributions_sample_count.append(len(train_y))
            test_target_distributions.append(test_target_distribution)
            test_target_distributions_sample_count.append(len(test_y))

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
    @property
    def target_names(self) -> list[str] | None:
        """In multi-output regression, the target names of the dataset."""
        return self.__target_names

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def target_range(self) -> list[float] | list[list[float]] | None:
        """Per-output ``[min, max]`` pairs (multioutput) or flat ``[min, max]``.

        In multi-output regression, one ``[min, max]`` pair per output, aligned
        with ``target_names``. In single-output regression, the flat
        ``[min, max]`` of the target. ``None`` in classification.
        """
        if self.__classes or (self.report.y is None):
            return None

        if self.__target_ranges is not None:  # multioutput regression
            return self.__target_ranges

        target = cast(np.ndarray, self.report.y)

        return [float(target.min()), float(target.max())]

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
            for report in self.report.reports_
        ]

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def metrics(self) -> list[Metric[CrossValidationReport]]:
        """
        The list of scalar metrics that have been computed from the report.

        Notes
        -----
        Per-label (per-class) and per-output (multioutput regression) metrics are
        aggregated independently for each label/output and sent with their
        dimension so the UI can expose a toggle. Metrics aggregated across labels
        or outputs are aggregated independently for each ``average`` mode and sent
        with their ``average`` dimension so the UI can show them as the aggregate
        variant, except for binary classification where only per-label rows are
        sent (``average`` is always ``None``). Only non-scalar values (``NaN``)
        are ignored.
        """
        data = self.report.metrics.summarize(data_source="both").summary
        selected = data[data["score"].notna()]
        if self.report._ml_task == "binary-classification":
            selected = selected[selected["average"].isna()]

        aggregated = (
            selected.groupby(
                [
                    "name",
                    "verbose_name",
                    "data_source",
                    "greater_is_better",
                    "label",
                    "output",
                    "average",
                ],
                dropna=False,
            )
            .agg(
                mean=("score", "mean"),
                std=("score", lambda s: s.std() if len(s) > 1 else 0.0),
            )
            .reset_index()
        )

        return [
            Metric(
                name=f"{row['name']}_{suffix}",
                verbose_name=f"{row['verbose_name']} - {suffix.upper()}",
                data_source=row["data_source"],
                greater_is_better=row["greater_is_better"]
                if suffix == "mean"
                else False,
                value=row[suffix],
                label=None if pd.isna(row["label"]) else row["label"],
                output=None if pd.isna(row["output"]) else int(row["output"]),
                average=None if pd.isna(row["average"]) else row["average"],
            )
            for row in aggregated.to_dict("records")
            for suffix in ("mean", "std")
        ]
