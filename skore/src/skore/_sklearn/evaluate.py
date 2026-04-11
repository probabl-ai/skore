"""Implement evaluate."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from skore._sklearn._comparison.report import ComparisonReport
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._estimator.report import EstimatorReport
from skore._sklearn.train_test_split.train_test_split import TrainTestSplit

if TYPE_CHECKING:
    from collections.abc import Generator

    from skore._sklearn.types import SKLearnCrossValidator


def evaluate(
    estimator: BaseEstimator | list[BaseEstimator] | dict[str, BaseEstimator],
    X: ArrayLike | list[ArrayLike | None] | dict[str, ArrayLike | None] | None = None,
    y: ArrayLike | None = None,
    data: dict | None = None,
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
    estimator : estimator object, list of estimators, or dict of estimators
        A scikit-learn compatible estimator; a list of estimators to compare; or a
        mapping of names to estimators.

    X : array-like, list of array-like, dict of array-like, or None
        Feature matrix. When ``estimator`` is a list, ``X`` can be a list of
        feature matrices (one per estimator) to compare models with different
        preprocessing pipelines. When ``estimator`` is a dict, ``X`` can be a
        dict with the **same keys** mapping each name to its feature matrix, or
        a single matrix broadcast to every estimator. A list of ``X`` is not
        supported when ``estimator`` is a dict; use a dict aligned on names or a
        single matrix.

    y : array-like of shape (n_samples,)
        Target vector.

    splitter : float, int, str, or cross-validation object, default=0.2
        Determines how the data is split:

        - ``float``: perform a single train-test split where the data is shuffled before
          splitting with a fixed seed (``random_state=0``) for reproducibility.
          Pass a :class:`~skore.TrainTestSplit` instance for more control over the
          splitting parameters.
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
    report : :class:`~skore.EstimatorReport`, :class:`~skore.CrossValidationReport` \
        or :class:`~skore.ComparisonReport`
        The report corresponding to the evaluation strategy.

    Raises
    ------
    ValueError
        If ``splitter`` is a string other than ``"prefit"``, or if ``estimator``
        is a dict and ``X`` is a dict whose keys do not match those of
        ``estimator``.

    TypeError
        If ``estimator`` is a dict and ``X`` is a list, or if ``estimator`` is a
        list and ``X`` is a dict.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import evaluate
    >>> X, y = make_classification(random_state=42)

    Default 80/20 train-test split:

    >>> report = evaluate(LogisticRegression(), X, y)

    Cross-validation with 5 folds:

    >>> report = evaluate(LogisticRegression(), X, y, splitter=5)

    Evaluate a pre-fitted estimator:

    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> fitted_model = LogisticRegression().fit(X_train, y_train)
    >>> report = evaluate(fitted_model, X_test, y_test, splitter="prefit")

    Compare several named estimators:

    >>> report = evaluate(
    ...     {"m1": LogisticRegression(), "m2": LogisticRegression(C=2.0)},
    ...     X,
    ...     y,
    ...     splitter=0.2,
    ... )
    >>> sorted(report.reports_)
    ['m1', 'm2']
    """
    if isinstance(estimator, (list, dict)):
        if isinstance(splitter, float):
            splitter = TrainTestSplit(test_size=splitter)

        if isinstance(estimator, dict):
            names, estimator = zip(*estimator.items(), strict=True)
            if isinstance(X, list):
                raise TypeError(
                    "When estimator is a dict, X cannot be a list. Pass a single "
                    "array-like broadcast to all estimators, or a "
                    "dict[str, array-like] with the same keys as estimator."
                )
            if isinstance(X, dict):
                if set(X) != set(names):
                    raise ValueError(
                        "When estimator and X are both dicts, they must have the "
                        f"same keys; got estimator keys {sorted(names)!r}"
                        f" and X keys {sorted(X)!r}."
                    )
                X = [X[name] for name in names]
            else:
                X = [X] * len(estimator)
        else:
            names = None
            if isinstance(X, dict):
                raise TypeError(
                    "When estimator is a list, X cannot be a dict. Pass a single "
                    "array-like broadcast to all estimators, or a list of "
                    "array-like with one matrix per estimator."
                )
            if not isinstance(X, list):
                X = [X] * len(estimator)

        reports = [
            evaluate(
                est,
                x,
                y,
                data=data,
                splitter=splitter,
                pos_label=pos_label,
                n_jobs=n_jobs,
            )
            for est, x in zip(estimator, X, strict=True)
        ]

        if names is not None:
            return ComparisonReport(
                cast(
                    dict[str, EstimatorReport] | dict[str, CrossValidationReport],
                    dict(zip(names, reports, strict=True)),
                ),
                n_jobs=n_jobs,
            )
        return ComparisonReport(
            cast(
                list[EstimatorReport] | list[CrossValidationReport],
                reports,
            ),
            n_jobs=n_jobs,
        )

    if isinstance(splitter, str):
        if splitter != "prefit":
            raise ValueError(
                f"Invalid string value for splitter: {splitter!r}. "
                "The only supported string value is 'prefit'."
            )
        return EstimatorReport(
            estimator,
            X_test=cast(ArrayLike | None, X),
            y_test=y,
            test_data=data,
            pos_label=pos_label,
        )

    if isinstance(splitter, float):
        splitter = TrainTestSplit(test_size=splitter)

    report = CrossValidationReport(
        estimator,
        cast(ArrayLike | None, X),
        y,
        data=data,
        pos_label=pos_label,
        splitter=splitter,
        n_jobs=n_jobs,
    )
    if len(report.estimator_reports_) == 1:
        return report.estimator_reports_[0]
    return report
