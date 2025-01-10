import joblib
from rich.progress import track
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv

from skore.externals._pandas_accessors import DirNamesMixin
from skore.externals._sklearn_compat import _safe_indexing
from skore.sklearn._base import _BaseReport
from skore.sklearn._estimator.report import EstimatorReport
from skore.sklearn.find_ml_task import _find_ml_task


def _generate_estimator_report(estimator, X, y, train_indices, test_indices):
    return EstimatorReport(
        estimator,
        fit=True,
        X_train=_safe_indexing(X, train_indices),
        y_train=_safe_indexing(y, train_indices),
        X_test=_safe_indexing(X, test_indices),
        y_test=_safe_indexing(y, test_indices),
    )


class CrossValidationReport(_BaseReport, DirNamesMixin):
    """Reporter for cross-validation results.

    Parameters
    ----------
    estimator : estimator object
        Estimator to make report from.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of supervised learning.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    """

    _ACCESSOR_CONFIG = {
        # "metrics": {"icon": ":straight_ruler:", "name": "metrics"},
    }

    def __init__(
        self,
        estimator,
        X,
        y=None,
        cv=None,
        n_jobs=None,
    ):
        cv = check_cv(cv, y, classifier=is_classifier(estimator))
        parallel = joblib.Parallel(n_jobs=n_jobs, return_as="generator_unordered")
        # do not split the data to take advantage of the memory mapping
        generator = parallel(
            joblib.delayed(_generate_estimator_report)(
                estimator,
                X,
                y,
                train_indices,
                test_indices,
            )
            for train_indices, test_indices in cv.split(X, y)
        )

        n_splits = cv.get_n_splits(X, y)

        self.cv_results = list(
            track(generator, total=n_splits, description="Processing cross-validation")
        )
        self._ml_task = _find_ml_task(y, estimator=self.cv_results[0].estimator)

    ####################################################################################
    # Methods related to the help and repr
    ####################################################################################

    def _get_help_panel_title(self):
        return (
            f"[bold cyan]📓 Tools to diagnose estimator "
            f"{self.estimator_name}[/bold cyan]"
        )

    def _get_help_legend(self):
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )
