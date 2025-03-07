from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Literal, Optional, Union, cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn import metrics
from sklearn.base import is_classifier, is_regressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor
from skore.sklearn._estimator.report import EstimatorReport
from skore.utils._accessor import _check_has_coef, _check_has_feature_importances
from skore.utils._index import flatten_multi_index

DataSource = Literal["test", "train", "X_y"]

Metric = Literal[
    "accuracy",
    "precision",
    "recall",
    "brier_score",
    "roc_auc",
    "log_loss",
    "r2",
    "rmse",
]

# If scoring represents a single score, one can use:
#   - a single string (see The scoring parameter: defining model evaluation rules);
#   - a callable (see Callable scorers) that returns a single value.
# If scoring represents multiple scores, one can use:
#   - a list or tuple of unique strings;
#   - a callable returning a dictionary where the keys are the metric names
#   and the values are the metric scores;
#   - a dictionary with metric names as keys and callables a values.
Scoring = Union[Metric, Callable, Iterable[Metric], dict[str, Callable]]

metric_to_scorer: dict[Metric, Callable] = {
    "accuracy": make_scorer(metrics.accuracy_score),
    "precision": make_scorer(metrics.precision_score),
    "recall": make_scorer(metrics.recall_score),
    "brier_score": make_scorer(metrics.brier_score_loss),
    "roc_auc": make_scorer(metrics.roc_auc_score),
    "log_loss": make_scorer(metrics.log_loss),
    "r2": make_scorer(metrics.r2_score),
    "rmse": make_scorer(metrics.root_mean_squared_error),
}


def _check_scoring(scoring: Any) -> Union[Scoring, None]:
    """Check that `scoring` is valid, and convert it to a suitable form as needed.

    If `scoring` is a list of strings, it is checked against our own metric names.
    For example, "rmse" is recognized as root-mean-square error, even though sklearn
    itself does not recognize this name.
    Similarly, "neg_root_mean_square_error" is not recognized, and leads to an error.

    Parameters
    ----------
    scoring : str, callable, list, tuple, dict, or None
        The scoring to check.

    Returns
    -------
    scoring
        A scoring hopefully suitable for passing to `permutation_importance`.
        Can be equal to the original scoring.

    Raises
    ------
    TypeError
        If `scoring` does not type-check.

    Examples
    --------
    >>> from sklearn.metrics import make_scorer, root_mean_squared_error
    >>> from skore.sklearn._estimator.feature_importance_accessor import _check_scoring

    >>> _check_scoring(None)  # Returns None

    >>> _check_scoring(make_scorer(root_mean_squared_error))
    make_scorer(root_mean_squared_error, ...)

    >>> _check_scoring({"rmse": make_scorer(root_mean_squared_error)})
    {'rmse': make_scorer(root_mean_squared_error, ...)}

    >>> _check_scoring("rmse")
    {'rmse': make_scorer(root_mean_squared_error, ...)}

    >>> _check_scoring(["r2", "rmse"])
    {'r2': make_scorer(r2_score, ...),
    'rmse': make_scorer(root_mean_squared_error, ...)}

    >>> _check_scoring("neg_root_mean_squared_error")
    Traceback (most recent call last):
    TypeError: If scoring is a string, it must be one of ...;
    got 'neg_root_mean_squared_error'

    >>> _check_scoring(["r2", make_scorer(root_mean_squared_error)])
    Traceback (most recent call last):
    TypeError: If scoring is a list or tuple, it must contain only strings; ...

    >>> _check_scoring(3)
    Traceback (most recent call last):
    TypeError: scoring must be a string, callable, list, tuple or dict;
    got <class 'int'>
    """
    if scoring is None or callable(scoring) or isinstance(scoring, dict):
        return scoring
    elif isinstance(scoring, str):
        if scoring in metric_to_scorer:
            # Convert to scorer
            return {scoring: metric_to_scorer[cast(Metric, scoring)]}
        raise TypeError(
            "If scoring is a string, it must be one of "
            f"{list(metric_to_scorer.keys())}; got '{scoring}'"
        )
    elif isinstance(scoring, (list, tuple)):
        result: dict[str, Callable] = {}
        for s in scoring:
            if isinstance(s, str):
                result |= cast(dict[str, Callable], _check_scoring(s))
            else:
                raise TypeError(
                    "If scoring is a list or tuple, it must contain only strings; "
                    f"got {s} of type {type(s)}"
                )
        return result
    else:
        raise TypeError(
            "scoring must be a string, callable, list, tuple or dict; "
            f"got {type(scoring)}"
        )


Aggregation = Literal["mean", "std"]


class _FeatureImportanceAccessor(_BaseAccessor[EstimatorReport], DirNamesMixin):
    """Accessor for feature importance related operations.

    You can access this accessor using the `feature_importance` attribute.
    """

    def __init__(self, parent: EstimatorReport) -> None:
        super().__init__(parent)

    @available_if(_check_has_coef())
    def coefficients(self) -> pd.DataFrame:
        """Retrieve the coefficients of a linear model, including the intercept.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_diabetes(return_X_y=True), random_state=0
        ... )
        >>> regressor = Ridge()
        >>> report = EstimatorReport(
        ...     regressor,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> report.feature_importance.coefficients()
                   Coefficient
        Intercept     152.4...
        Feature #0     21.2...
        Feature #1    -60.4...
        Feature #2    302.8...
        Feature #3    179.4...
        Feature #4      8.9...
        Feature #5    -28.8...
        Feature #6   -149.3...
        Feature #7    112.6...
        Feature #8    250.5...
        Feature #9     99.5...
        """
        parent_estimator = self._parent.estimator_

        if isinstance(parent_estimator, Pipeline):
            feature_names = parent_estimator[:-1].get_feature_names_out()
        else:
            if hasattr(parent_estimator, "feature_names_in_"):
                feature_names = parent_estimator.feature_names_in_
            else:
                feature_names = [
                    f"Feature #{i}" for i in range(parent_estimator.n_features_in_)
                ]

        estimator = (
            parent_estimator[-1]
            if isinstance(parent_estimator, Pipeline)
            else parent_estimator
        )
        intercept = np.atleast_2d(estimator.intercept_)
        coef = np.atleast_2d(estimator.coef_)

        data = np.concatenate([intercept, coef.T])

        if data.shape[1] == 1:
            columns = ["Coefficient"]
        elif is_classifier(parent_estimator):
            columns = [f"Class #{i}" for i in range(data.shape[1])]
        else:
            columns = [f"Target #{i}" for i in range(data.shape[1])]

        df = pd.DataFrame(
            data=data,
            index=["Intercept"] + list(feature_names),
            columns=columns,
        )

        return df

    @available_if(_check_has_feature_importances())
    def mean_decrease_impurity(self):
        """Retrieve the mean decrease impurity (MDI) of a tree-based model.

        This method is available for estimators that expose a `feature_importances_`
        attribute. See for example
        :attr:`sklearn.ensemble.GradientBoostingClassifier.feature_importances_`.
        In particular, note that the MDI is computed at fit time, i.e. using the
        training data.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = make_classification(n_features=5, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        >>> forest = RandomForestClassifier(n_estimators=5, random_state=0)
        >>> report = EstimatorReport(
        ...     forest,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> report.feature_importance.mean_decrease_impurity()
                   Mean decrease impurity
        Feature #0                0.06...
        Feature #1                0.19...
        Feature #2                0.01...
        Feature #3                0.69...
        Feature #4                0.02...
        """
        parent_estimator = self._parent.estimator_
        estimator = (
            parent_estimator.steps[-1][1]
            if isinstance(parent_estimator, Pipeline)
            else parent_estimator
        )

        data = estimator.feature_importances_

        if isinstance(parent_estimator, Pipeline):
            feature_names = parent_estimator[:-1].get_feature_names_out()
        else:
            if hasattr(parent_estimator, "feature_names_in_"):
                feature_names = parent_estimator.feature_names_in_
            else:
                feature_names = [
                    f"Feature #{i}" for i in range(parent_estimator.n_features_in_)
                ]

        df = pd.DataFrame(
            data=data,
            index=feature_names,
            columns=["Mean decrease impurity"],
        )

        return df

    def feature_permutation(
        self,
        *,
        data_source: DataSource = "test",
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        aggregate: Optional[Union[Aggregation, list[Aggregation]]] = None,
        scoring: Optional[Scoring] = None,
        n_repeats: int = 5,
        max_samples: float = 1.0,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Report the permutation feature importance.

        This computes the permutation importance using sklearn's
        :func:`~sklearn.inspection.permutation_importance` function,
        which consists in permuting the values of one feature and comparing
        the value of `scoring` between with and without the permutation, which gives an
        indication on the impact of the feature.

        By default, `random_state` is set to `None`, which means the function will
        return a different result at every call. In that case, the results are not
        cached. If you wish to take advantage of skore's caching capabilities, make
        sure you set the `random_state` parameter.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the test
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the test
            target provided when creating the report.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the repeats.

        scoring : str, callable, list, tuple, or dict, default=None
            The scorer to pass to :func:`~sklearn.inspection.permutation_importance`.

            If `scoring` represents a single score, one can use:

            - a single string, which must be one of the supported metrics;
            - a callable that returns a single value.

            If `scoring` represents multiple scores, one can use:

            - a list or tuple of unique strings, which must be one of the supported
              metrics;
            - a callable returning a dictionary where the keys are the metric names
              and the values are the metric scores;
            - a dictionary with metric names as keys and callables a values.

        n_repeats : int, default=5
            Number of times to permute a feature.

        max_samples : int or float, default=1.0
            The number of samples to draw from `X` to compute feature importance
            in each repeat (without replacement).

            - If int, then draw max_samples samples.
            - If float, then draw max_samples * X.shape[0] samples.
            - If max_samples is equal to 1.0 or X.shape[0], all samples will be used.

            While using this option may provide less accurate importance estimates,
            it keeps the method tractable when evaluating feature importance on
            large datasets. In combination with n_repeats, this allows to control
            the computational speed vs statistical accuracy trade-off of this method.

        n_jobs : int or None, default=None
            Number of jobs to run in parallel. -1 means using all processors.

        random_state : int or None, default=None
            Pseudo-random number generator to control the permutations of each feature.
            Pass an int to get reproducible results across function calls.

        flat_index : bool, default=False
            Whether to flatten the multi-index columns. Flat index will always be lower
            case, do not include spaces and remove the hash symbol to ease indexing.

        Returns
        -------
        pandas.DataFrame
            The permutation importance.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *make_regression(n_features=3, random_state=0), random_state=0
        ... )
        >>> regressor = Ridge()
        >>> report = EstimatorReport(
        ...     regressor,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )

        >>> report.feature_importance.feature_permutation(
        ...    n_repeats=2,
        ...    random_state=0,
        ... )
        Repeat              Repeat #0  Repeat #1
        Metric  Feature
        r2      Feature #0   0.699...   0.885...
                Feature #1   2.320...   2.636...
                Feature #2   0.028...   0.022...

        >>> report.feature_importance.feature_permutation(
        ...    scoring=["r2", "rmse"],
        ...    n_repeats=2,
        ...    random_state=0,
        ... )
        Repeat             Repeat #0  Repeat #1
        Metric Feature
        r2     Feature #0   0.699...   0.885...
               Feature #1   2.320...   2.636...
               Feature #2   0.028...   0.022...
        rmse   Feature #0 -47.222... -53.231...
               Feature #1 -86.608... -92.366...
               Feature #2  -8.930...  -7.916...

        >>> report.feature_importance.feature_permutation(
        ...    n_repeats=2,
        ...    aggregate=["mean", "std"],
        ...    random_state=0,
        ... )
                                mean       std
        Metric  Feature
        r2      Feature #0  0.792...  0.131...
                Feature #1  2.478...  0.223...
                Feature #2  0.025...  0.003...

        >>> report.feature_importance.feature_permutation(
        ...    n_repeats=2,
        ...    aggregate=["mean", "std"],
        ...    flat_index=True,
        ...    random_state=0,
        ... )
                          mean       std
        r2_feature_0  0.792...  0.131...
        r2_feature_1  2.478...  0.223...
        r2_feature_2  0.025...  0.003...
        """
        return self._feature_permutation(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
            aggregate=aggregate,
            scoring=scoring,
            n_repeats=n_repeats,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state,
            flat_index=flat_index,
        )

    def _feature_permutation(
        self,
        *,
        data_source: DataSource = "test",
        data_source_hash: Optional[int] = None,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        aggregate: Optional[Union[Aggregation, list[Aggregation]]] = None,
        scoring: Optional[Scoring] = None,
        n_repeats: int = 5,
        max_samples: float = 1.0,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Private interface of `feature_permutation` to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around, or `None` and thus trigger its computation
        in the underlying process.
        """
        checked_scoring = _check_scoring(scoring)

        if data_source_hash is None:
            X_, y_true, data_source_hash = self._get_X_y_and_data_source_hash(
                data_source=data_source, X=X, y=y
            )

        # If random_state is None, then we do not do any caching
        if random_state is None:
            cache_key = None

        elif isinstance(random_state, int):
            # build the cache key components to finally create a tuple that will be used
            # to check if the metric has already been computed
            cache_key_parts: list[Any] = [
                self._parent._hash,
                "permutation_importance",
                data_source,
            ]

            if data_source_hash is not None:
                cache_key_parts.append(data_source_hash)

            if callable(scoring) or isinstance(scoring, (list, dict)):
                cache_key_parts.append(joblib.hash(scoring))
            else:
                cache_key_parts.append(scoring)

            # aggregate is not included in the cache
            # in order to trade off computation for storage

            # order arguments by key to ensure cache works
            # n_jobs variable should not be in the cache
            kwargs = {
                "n_repeats": n_repeats,
                "max_samples": max_samples,
                "random_state": random_state,
            }
            for _, v in sorted(kwargs.items()):
                cache_key_parts.append(v)

            cache_key = tuple(cache_key_parts)

        else:
            raise ValueError(
                f"random_state must be an integer or None; got {type(random_state)}"
            )

        if cache_key in self._parent._cache:
            score = self._parent._cache[cache_key]
        else:
            sklearn_score = permutation_importance(
                estimator=self._parent.estimator_,
                X=X_,
                y=y_true,
                scoring=checked_scoring,
                n_repeats=n_repeats,
                n_jobs=n_jobs,
                random_state=random_state,
                max_samples=max_samples,
            )
            score = sklearn_score.get("importances")

            feature_names = (
                self._parent.estimator_.feature_names_in_
                if hasattr(self._parent.estimator_, "feature_names_in_")
                else [f"Feature #{i}" for i in range(X_.shape[1])]
            )

            # If there is more than one metric
            if score is None:
                data = np.concatenate(
                    [v["importances"] for v in sklearn_score.values()]
                )
                n_repeats = data.shape[1]
                index = pd.MultiIndex.from_product(
                    [sklearn_score, feature_names], names=("Metric", "Feature")
                )
                columns = pd.Index(
                    (f"Repeat #{i}" for i in range(n_repeats)), name="Repeat"
                )
                score = pd.DataFrame(data=data, index=index, columns=columns)
            else:
                data = score
                n_repeats = data.shape[1]

                # Get score name
                if scoring is None:
                    if is_classifier(self._parent.estimator_):
                        scoring_name = "accuracy"
                    elif is_regressor(self._parent.estimator_):
                        scoring_name = "r2"
                else:
                    # e.g. if scoring is a callable
                    scoring_name = None

                    # no other cases to deal with explicitly, because
                    # scoring cannot possibly be a string at this stage

                if scoring_name is None:
                    index = pd.Index(feature_names, name="Feature")
                else:
                    index = pd.MultiIndex.from_product(
                        [[scoring_name], feature_names], names=("Metric", "Feature")
                    )

                columns = pd.Index(
                    (f"Repeat #{i}" for i in range(n_repeats)), name="Repeat"
                )
                score = pd.DataFrame(data=data, index=index, columns=columns)

            # Unless random_state is an int (i.e. the call is deterministic),
            # we do not cache
            if cache_key is not None:
                self._parent._cache[cache_key] = score

        if aggregate:
            if isinstance(aggregate, str):
                aggregate = [aggregate]
            score = score.aggregate(func=aggregate, axis=1)

        if flat_index and isinstance(score.index, pd.MultiIndex):
            score.index = flatten_multi_index(score.index)

        return score

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _format_method_name(self, name: str) -> str:
        return f"{name}(...)".ljust(29)

    def _get_help_panel_title(self) -> str:
        return "[bold cyan]Available feature importance methods[/bold cyan]"

    def _get_help_tree_title(self) -> str:
        return "[bold cyan]report.feature_importance[/bold cyan]"

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(
            class_name="skore.EstimatorReport.feature_importance",
            help_method_name="report.feature_importance.help()",
        )
