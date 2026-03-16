"""Dispatch function to compare reports."""

from skore._sklearn._comparison.report import ComparisonReport
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._estimator.report import EstimatorReport


def compare(
    reports: (
        list[EstimatorReport]
        | dict[str, EstimatorReport]
        | list[CrossValidationReport]
        | dict[str, CrossValidationReport]
    ),
    *,
    n_jobs: int | None = None,
) -> ComparisonReport:
    """Consolidate reports into a single :class:`~skore.ComparisonReport`.

    Parameters
    ----------
    reports : list of reports or dict
        Reports to compare. If a dict, keys will be used to label the estimators;
        if a list, the labels are computed from the estimator class names.
        Expects at least two reports to compare, with the same test target.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimators and computing
        the scores are parallelized.
        When accessing some methods of the `ComparisonReport`, the `n_jobs`
        parameter is used to parallelize the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Returns
    -------
    :class:`~skore.ComparisonReport`
        A comparison report containing the reports to compare.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import evaluate, compare
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> estimator_1 = LogisticRegression()
    >>> estimator_2 = LogisticRegression(C=2)
    >>> report_1 = evaluate(estimator_1, X, y, pos_label=1, splitter=0.2)
    >>> report_2 = evaluate(estimator_2, X, y, pos_label=1, splitter=0.2)
    >>> report = compare([report_1, report_2])
    """
    return ComparisonReport(reports, n_jobs=n_jobs)
