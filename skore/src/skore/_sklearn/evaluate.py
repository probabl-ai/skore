"""Implement evaluate."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from skore._sklearn._comparison.report import ComparisonReport
    from skore._sklearn._cross_validation.report import CrossValidationReport
    from skore._sklearn._estimator.report import EstimatorReport
    from skore._sklearn.types import SKLearnCrossValidator


def evaluate(
    estimator: BaseEstimator | list[BaseEstimator],
    X: ArrayLike,
    y: ArrayLike | None = None,
    *,
    splitter: float | int | str | SKLearnCrossValidator | Generator | None = 0.2,
    pos_label: int | float | bool | str | None = None,
    n_jobs: int | None = None,
) -> EstimatorReport | CrossValidationReport | ComparisonReport:
    """Evaluate one or more estimators on the given data.

    This function is a dispatcher that creates the appropriate report based on
    the ``splitter`` parameter:

    - A float triggers a train-test split and returns an
      :class:`~skore.EstimatorReport`.
    - The string ``"prefit"`` assumes the estimator is already fitted and
      evaluates it on ``X`` and ``y`` as test data.
    - An integer or cross-validation object triggers cross-validation and
      returns a :class:`~skore.CrossValidationReport`.
    - When ``estimator`` is a list, each estimator is evaluated individually
      and the results are wrapped in a :class:`~skore.ComparisonReport`.

    Parameters
    ----------
    estimator : estimator object or list of estimator objects
        A scikit-learn compatible estimator, or a list of estimators to compare.

    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,) or None, default=None
        Target vector. Can be ``None`` for unsupervised tasks.

    splitter : float, int, str, cross-validation object, or None, default=0.2
        Determines how the data is split:

        - ``float``: fraction used as ``test_size`` in a train-test split
          (e.g. ``0.2`` means 80% train / 20% test).
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
        If ``splitter="prefit"`` and the estimator is not fitted.
    TypeError
        If ``splitter`` is not a recognized type.

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
        return _evaluate_multiple(
            estimator, X, y, splitter=splitter, pos_label=pos_label, n_jobs=n_jobs
        )

    return _evaluate_single(
        estimator, X, y, splitter=splitter, pos_label=pos_label, n_jobs=n_jobs
    )


def _evaluate_single(
    estimator: BaseEstimator,
    X: Any,
    y: Any,
    *,
    splitter: Any,
    pos_label: Any,
    n_jobs: Any,
) -> EstimatorReport | CrossValidationReport:
    """Evaluate a single estimator based on the splitter type."""
    from skore._sklearn._cross_validation.report import CrossValidationReport
    from skore._sklearn._estimator.report import EstimatorReport
    from skore._sklearn.train_test_split.train_test_split import train_test_split

    if isinstance(splitter, str):
        if splitter != "prefit":
            raise ValueError(
                f"Invalid string value for splitter: {splitter!r}. "
                "The only supported string value is 'prefit'."
            )
        check_is_fitted(estimator)
        return EstimatorReport(
            estimator, fit=False, X_test=X, y_test=y, pos_label=pos_label
        )

    if isinstance(splitter, (float, type(None))):
        test_size = splitter if splitter is not None else 0.2
        split_data = train_test_split(X, y, test_size=test_size, as_dict=True)
        return EstimatorReport(
            estimator,
            fit=True,
            X_train=split_data["X_train"],
            y_train=split_data["y_train"],
            X_test=split_data["X_test"],
            y_test=split_data["y_test"],
            pos_label=pos_label,
        )

    if isinstance(splitter, int):
        return CrossValidationReport(
            estimator, X, y, pos_label=pos_label, splitter=splitter, n_jobs=n_jobs
        )

    # Duck-type check for CV splitter objects (has split and get_n_splits)
    if hasattr(splitter, "split") and hasattr(splitter, "get_n_splits"):
        return CrossValidationReport(
            estimator, X, y, pos_label=pos_label, splitter=splitter, n_jobs=n_jobs
        )

    raise TypeError(
        f"Invalid type for splitter: {type(splitter).__name__!r}. Expected a float, "
        "int, 'prefit', or a cross-validation object with 'split' and "
        "'get_n_splits' methods."
    )


def _evaluate_multiple(
    estimators: list[BaseEstimator],
    X: Any,
    y: Any,
    *,
    splitter: Any,
    pos_label: Any,
    n_jobs: Any,
) -> ComparisonReport:
    """Evaluate multiple estimators and wrap results in a ComparisonReport."""
    from skore._sklearn._comparison.report import ComparisonReport
    from skore._sklearn._estimator.report import EstimatorReport
    from skore._sklearn.train_test_split.train_test_split import train_test_split

    # For float splitters, split once and reuse the same train/test data for all
    # estimators so that ComparisonReport sees identical test targets.
    if isinstance(splitter, (float, type(None))):
        test_size = splitter if splitter is not None else 0.2
        split_data = train_test_split(X, y, test_size=test_size, as_dict=True)
        reports = [
            EstimatorReport(
                est,
                fit=True,
                X_train=split_data["X_train"],
                y_train=split_data["y_train"],
                X_test=split_data["X_test"],
                y_test=split_data["y_test"],
                pos_label=pos_label,
            )
            for est in estimators
        ]
    else:
        reports = [
            _evaluate_single(
                est, X, y, splitter=splitter, pos_label=pos_label, n_jobs=n_jobs
            )
            for est in estimators
        ]

    return ComparisonReport(reports, n_jobs=n_jobs)
