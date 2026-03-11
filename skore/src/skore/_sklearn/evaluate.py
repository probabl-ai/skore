"""Implement evaluate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from skore._sklearn._comparison.report import ComparisonReport
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._estimator.report import EstimatorReport

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import ArrayLike

    from skore._sklearn.types import SKLearnCrossValidator


class _TrainTestSplit:
    """Single train-test split implementing the cross-validation protocol.

    This private splitter wraps ``sklearn.model_selection.train_test_split`` and
    exposes ``split`` / ``get_n_splits`` so that it can be passed to
    :class:`~skore.CrossValidationReport`.
    """

    def __init__(
        self,
        test_size: float | int | None = 0.2,
        train_size: float | int | None = None,
        random_state: int | np.random.RandomState | None = 0,
        shuffle: bool = True,
        stratify: ArrayLike | None = None,
    ) -> None:
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of splits (always 1)."""
        return 1

    def split(self, X: Any, y: Any = None, groups: Any = None):
        """Generate a single train-test split of indices."""
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        indices = np.arange(n_samples)
        train_idx, test_idx = sklearn_train_test_split(
            indices,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )
        yield (train_idx, test_idx)


def evaluate(
    estimator: BaseEstimator | list[BaseEstimator],
    X: ArrayLike | list[ArrayLike],
    y: ArrayLike | None = None,
    *,
    splitter: float | int | str | SKLearnCrossValidator | Generator = 0.2,
    pos_label: int | float | bool | str | None = None,
    n_jobs: int | None = None,
) -> EstimatorReport | CrossValidationReport | ComparisonReport:
    """Evaluate one or more estimators on the given data.

    Passing several estimators provides a report to compare them, while the
    ``splitter`` parameter controls whether a train-test split or
    cross-validation is used.

    Parameters
    ----------
    estimator : estimator object or list of estimator objects
        A scikit-learn compatible estimator, or a list of such estimators
        to compare.

    X : array-like of shape (n_samples, n_features) or list of array-like
        Feature matrix. When ``estimator`` is a list, ``X`` can also be a
        list of feature matrices (one per estimator) to assess the impact
        of different feature engineering pipelines.

    y : array-like of shape (n_samples,) or None, default=None
        Target vector. Can be ``None`` for unsupervised tasks.

    splitter : float, int, str, or cross-validation object, default=0.2
        Determines how the data is split:

        - ``float``: fraction used as ``test_size`` in a single train-test
          split (e.g. ``0.2`` means 80% train / 20% test).  The data is
          shuffled before splitting with a fixed seed
          (``random_state=0``) for reproducibility.
        - ``"prefit"``: the estimator is assumed to be already fitted; ``X``
          and ``y`` are used as the test set.
        - ``int``: number of folds for cross-validation (passed to
          :class:`~skore.CrossValidationReport`).
        - cross-validation splitter (e.g. ``KFold``, ``StratifiedKFold``):
          passed directly to :class:`~skore.CrossValidationReport`.

    pos_label : int, float, bool or str, default=None
        The positive class label for binary classification metrics. Forwarded
        to the underlying report.

    n_jobs : int or None, default=None
        Number of jobs for parallel execution. Forwarded to
        :class:`~skore.CrossValidationReport` or
        :class:`~skore.ComparisonReport`.

    Returns
    -------
    report : EstimatorReport, CrossValidationReport, or ComparisonReport
        The report corresponding to the evaluation strategy.

    Raises
    ------
    ValueError
        If ``splitter`` is a string other than ``"prefit"``.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import evaluate
    >>> X, y = make_classification(random_state=42)
    >>> report = evaluate(LogisticRegression(), X, y)

    Cross-validation with 5 folds:

    >>> report = evaluate(LogisticRegression(), X, y, splitter=5)

    Evaluate a pre-fitted estimator:

    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> fitted_model = LogisticRegression().fit(X_train, y_train)
    >>> report = evaluate(fitted_model, X_test, y_test, splitter="prefit")
    """
    if isinstance(estimator, list):
        if isinstance(splitter, float):
            splitter = _TrainTestSplit(test_size=splitter)

        if not isinstance(X, list):
            X = [X] * len(estimator)
        reports = [
            evaluate(
                est,
                x,
                y,
                splitter=splitter,
                pos_label=pos_label,
                n_jobs=n_jobs,
            )
            for est, x in zip(estimator, X, strict=True)
        ]
        return ComparisonReport(
            cast(
                list[EstimatorReport] | list[CrossValidationReport],
                reports,
            ),
            n_jobs=n_jobs,
        )

    if isinstance(X, list):
        raise TypeError("X must be a single array-like when estimator is not a list.")

    if isinstance(splitter, str):
        if splitter != "prefit":
            raise ValueError(
                f"Invalid string value for splitter: {splitter!r}. "
                "The only supported string value is 'prefit'."
            )
        return EstimatorReport(estimator, X_test=X, y_test=y, pos_label=pos_label)

    if isinstance(splitter, float):
        splitter = _TrainTestSplit(test_size=splitter)

    report = CrossValidationReport(
        estimator, X, y, pos_label=pos_label, splitter=splitter, n_jobs=n_jobs
    )
    if hasattr(splitter, "get_n_splits") and splitter.get_n_splits() == 1:
        return report.estimator_reports_[0]
    return report
