from collections.abc import Iterable
from typing import Any, Callable, Literal, Optional, Union

import joblib
import numpy as np
import pandas as pd
from numpy.random import RandomState
from numpy.typing import ArrayLike
from sklearn.base import is_classifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor
from skore.sklearn._estimator.report import EstimatorReport
from skore.utils._accessor import _check_has_coef

DataSource = Literal["test", "train", "X_y"]

# If scoring represents a single score, one can use:
#   - a single string (see The scoring parameter: defining model evaluation rules);
#   - a callable (see Callable scorers) that returns a single value.
# If scoring represents multiple scores, one can use:
#   - a list or tuple of unique strings;
#   - a callable returning a dictionary where the keys are the metric names
#   and the values are the metric scores;
#   - a dictionary with metric names as keys and callables a values.
Scoring = Union[str, Callable, Iterable[str], dict[str, Callable]]


class _FeatureImportanceAccessor(_BaseAccessor["EstimatorReport"], DirNamesMixin):
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
        estimator = self._parent.estimator_

        if isinstance(estimator, Pipeline):
            feature_names = estimator[:-1].get_feature_names_out()
        else:
            if hasattr(estimator, "feature_names_in_"):
                feature_names = estimator.feature_names_in_
            else:
                feature_names = [
                    f"Feature #{i}" for i in range(estimator.n_features_in_)
                ]

        linear_model = estimator[-1] if isinstance(estimator, Pipeline) else estimator
        intercept = np.atleast_2d(linear_model.intercept_)
        coef = np.atleast_2d(linear_model.coef_)

        data = np.concatenate([intercept, coef.T])

        if data.shape[1] == 1:
            columns = ["Coefficient"]
        elif is_classifier(estimator):
            columns = [f"Class #{i}" for i in range(data.shape[1])]
        else:
            columns = [f"Target #{i}" for i in range(data.shape[1])]

        df = pd.DataFrame(
            data=data,
            index=["Intercept"] + list(feature_names),
            columns=columns,
        )

        return df

    def feature_permutation(
        self,
        *,
        data_source: DataSource = "test",
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        scoring: Optional[Scoring] = None,
        n_repeats: int = 5,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, RandomState]] = None,
        max_samples: float = 1.0,
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

        scoring : str, callable, list, tuple, or dict, default=None
            The scorer to pass to :func:`~sklearn.inspection.permutation_importance`.

        n_repeats : int, default=5
            Number of times to permute a feature.

        n_jobs : int or None, default=None
            Number of jobs to run in parallel. -1 means using all processors.

        random_state : int, RandomState instance, default=None
            Pseudo-random number generator to control the permutations of each feature.
            Pass an int to get reproducible results across function calls.

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

        Returns
        -------
        pandas.DataFrame
            The permutation importance.
        """
        return self._feature_permutation(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
            scoring=scoring,
            n_repeats=n_repeats,
            n_jobs=n_jobs,
            random_state=random_state,
            max_samples=max_samples,
        )

    def _feature_permutation(
        self,
        *,
        data_source: DataSource = "test",
        data_source_hash: Optional[int] = None,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        scoring: Optional[Scoring] = None,
        n_repeats: int = 5,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, RandomState]] = None,
        max_samples: float = 1.0,
    ) -> pd.DataFrame:
        """Private interface of `feature_permutation` to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around, or `None` and thus trigger its computation
        in the underlying process.
        """
        if data_source_hash is None:
            X_, y_true, data_source_hash = self._get_X_y_and_data_source_hash(
                data_source=data_source, X=X, y=y
            )

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

        # order arguments by key to ensure cache works
        # n_jobs variable should not be in the cache
        kwargs = {
            "n_repeats": n_repeats,
            "random_state": random_state,
            "max_samples": max_samples,
        }
        for _, v in sorted(kwargs.items()):
            cache_key_parts.append(v)

        cache_key = tuple(cache_key_parts)

        if cache_key in self._parent._cache:
            score = self._parent._cache[cache_key]
        else:
            sklearn_score = permutation_importance(
                estimator=self._parent.estimator_,
                X=X_,
                y=y_true,
                scoring=scoring,
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
                index = pd.Index(feature_names, name="Feature")
                columns = pd.Index(
                    (f"Repeat #{i}" for i in range(n_repeats)), name="Repeat"
                )
                score = pd.DataFrame(data=data, index=index, columns=columns)

            # Unless random_state is an int (i.e. the call is deterministic),
            # we do not cache
            if isinstance(kwargs.get("random_state"), int):
                self._parent._cache[cache_key] = score

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
