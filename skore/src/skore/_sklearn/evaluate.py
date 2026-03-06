"""Implement evaluate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    The split result is cached so that the same indices are reused when
    comparing multiple estimators.
    """

    def __init__(self, test_size: float = 0.2) -> None:
        self.test_size = test_size
        self._cached_split: tuple[np.ndarray, np.ndarray] | None = None

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of splits (always 1)."""
        return 1

    def split(self, X: Any, y: Any = None, groups: Any = None):
        """Generate a single train-test split of indices."""
        if self._cached_split is None:
            n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
            indices = np.arange(n_samples)
            train_idx, test_idx = sklearn_train_test_split(
                indices, test_size=self.test_size
            )
            self._cached_split = (train_idx, test_idx)
        yield self._cached_split


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
          split (e.g. ``0.2`` means 80% train / 20% test).
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

    >>> fitted_model = LogisticRegression().fit(X, y)
    >>> report = evaluate(fitted_model, X, y, splitter="prefit")
    """
    if isinstance(estimator, list):
        if isinstance(splitter, float):
            splitter = _TrainTestSplit(test_size=splitter)

        if isinstance(X, list):
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
        else:
            reports = [
                evaluate(
                    est,
                    X,
                    y,
                    splitter=splitter,
                    pos_label=pos_label,
                    n_jobs=n_jobs,
                )
                for est in estimator
            ]
        return ComparisonReport(reports, n_jobs=n_jobs)

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
