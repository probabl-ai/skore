"""Implement evaluate."""

from __future__ import annotations

from collections.abc import Generator
from typing import Literal, cast

from numpy.typing import ArrayLike

from skore import configuration
from skore._sklearn._comparison.report import ComparisonReport
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._estimator.report import EstimatorReport
from skore._sklearn.train_test_split import TrainTestSplit
from skore._sklearn.types import (
    _DEFAULT,
    EstimatorLike,
    SKLearnCrossValidator,
    _DefaultType,
)
from skore._utils._skrub import data_op_has_explicit_cv, get_data_op


def evaluate(
    estimator: EstimatorLike | list[EstimatorLike] | dict[str, EstimatorLike],
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    data: dict | None = None,
    *,
    splitter: float
    | int
    | Literal["prefit"]
    | SKLearnCrossValidator
    | Generator
    | _DefaultType = _DEFAULT,
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
        The estimator to evaluate of several estimators to compare. An estimator can
        be one of the following:

        - a scikit-learn compatible estimator as a :class:`~sklearn.base.BaseEstimator`;
        - a skrub :class:`~skrub.DataOp` to preprocess the data;
        - a skrub :class:`~skrub.SkrubLearner` extracted from a :class:`~skrub.DataOp`
          by calling :meth:`~skrub.DataOp.skb.make_learner`.

    X : array-like or None
        Feature matrix shared by all estimators when comparing several models.
        When comparing prefit estimators and no test features are needed,
        pass ``X=None``. To compare estimators evaluated on different feature
        matrices, call :func:`~skore.evaluate` once per estimator, then
        :func:`~skore.compare`.

    y : array-like of shape (n_samples,), or None
        Target vector.

    data : dict or None
        When ``estimator`` is a skrub :class:`~skrub.SkrubLearner`, bindings for
        variables contained in the DataOp that was used to create this learner
        (e.g. ``{"X": X_df, "other_table": df, ...}``).

    splitter : float, int, "prefit", or cross-validation object, default=0.2
        Determines how the data is split. When omitted, a skrub learner whose
        DataOp was configured with an explicit cross-validation splitter via
        :meth:`~skrub.DataOp.skb.mark_as_X` uses that splitter (including
        ``split_kwargs`` such as ``groups``). Otherwise, the default is a
        single 80/20 train-test split:

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

    See Also
    --------
    :func:`~skore.compare` :
        Compare already evaluated reports.
    :class:`~skore.EstimatorReport` :
        Report for a fitted estimator on a test set.
    :class:`~skore.CrossValidationReport` :
        Report for cross-validation of an estimator.
    :class:`~skore.ComparisonReport` :
        Report comparing several evaluated models.

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
    >>> list(report.reports_)
    ['m1', 'm2']
    """
    if isinstance(X, (list, dict)):
        raise TypeError(
            "X must be a single array-like or None. To compare estimators "
            "evaluated on different feature matrices, call evaluate() once per "
            "model, then use skore.compare()."
        )

    if X is None and y is None and data is None:
        raise ValueError(
            "Provide data through X and y or through data to evaluate your estimator."
        )

    if isinstance(estimator, (list, dict)):
        if isinstance(estimator, dict):
            names = list(estimator.keys())
            estimators = list(estimator.values())
        else:
            names = None
            estimators = estimator

        Xs = [cast(ArrayLike, X)] * len(estimators)

        reports = cast(
            list[EstimatorReport] | list[CrossValidationReport],
            [
                evaluate(
                    est,
                    x,
                    y,
                    data=data,
                    splitter=splitter,
                    pos_label=pos_label,
                    n_jobs=n_jobs,
                )
                for est, x in zip(estimators, Xs, strict=True)
            ],
        )

        if names is not None:
            return ComparisonReport(
                dict(zip(names, reports, strict=True)), n_jobs=n_jobs
            )
        return ComparisonReport(reports, n_jobs=n_jobs)

    if splitter is _DEFAULT:
        data_op = get_data_op(estimator)
        if data_op is not None and data_op_has_explicit_cv(data_op):
            return CrossValidationReport(
                estimator,
                X,
                y,
                data=data,
                pos_label=pos_label,
                splitter=None,
                n_jobs=n_jobs,
            )
        splitter = 0.2

    if isinstance(splitter, str):
        if splitter != "prefit":
            raise ValueError(
                f"Invalid string value for splitter: {splitter!r}. "
                "The only supported string value is 'prefit'."
            )
        return EstimatorReport(
            estimator,
            X_test=X,
            y_test=y,
            test_data=data,
            pos_label=pos_label,
        )

    if isinstance(splitter, float):
        splitter = TrainTestSplit(test_size=splitter)

    if isinstance(splitter, TrainTestSplit):
        # It's easier to make a 1-split CrossValidationReport
        # and extract an EstimatorReport from it,
        # than to make an EstimatorReport from scratch
        with configuration(show_progress=False):
            report = CrossValidationReport(
                estimator,
                X,
                y,
                data=data,
                pos_label=pos_label,
                splitter=splitter,
                n_jobs=n_jobs,
            )
        return report.reports_[0]

    splitter = cast(int | SKLearnCrossValidator | Generator, splitter)

    return CrossValidationReport(
        estimator,
        X,
        y,
        data=data,
        pos_label=pos_label,
        splitter=splitter,
        n_jobs=n_jobs,
    )
