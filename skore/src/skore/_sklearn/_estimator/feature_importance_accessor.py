from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal, cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.sparse import issparse
from sklearn import metrics
from sklearn.base import is_classifier, is_regressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._estimator.report import EstimatorReport
from skore._sklearn._plot.metrics.feature_importance_coefficients_display import (
    FeatureImportanceCoefficientsDisplay,
)
from skore._sklearn.types import Aggregate
from skore._utils._accessor import (
    _check_estimator_has_coef,
    _check_has_feature_importances,
)
from skore._utils._index import flatten_multi_index

DataSource = Literal["test", "train", "X_y"]


MetricNames = Literal[
    "accuracy",
    "precision",
    "recall",
    "brier_score",
    "roc_auc",
    "log_loss",
    "r2",
    "rmse",
]

# If the metric parameter represents a single metric, one can use:
#   - a single string (see The metric parameter: defining model evaluation rules);
#   - a callable (see Callable scorers) that returns a single value.
# If the metric parameter represents multiple metrics, one can use:
#   - a list or tuple of unique strings;
#   - a callable returning a dictionary where the keys are the metric names
#   and the values are the metric scores;
#   - a dictionary with metric names as keys and callables a values.
Metric = MetricNames | Callable | Iterable[MetricNames] | dict[str, Callable]

metric_to_scorer: dict[MetricNames, Callable] = {
    "accuracy": make_scorer(metrics.accuracy_score),
    "precision": make_scorer(metrics.precision_score),
    "recall": make_scorer(metrics.recall_score),
    "brier_score": make_scorer(metrics.brier_score_loss),
    "roc_auc": make_scorer(metrics.roc_auc_score),
    "log_loss": make_scorer(metrics.log_loss),
    "r2": make_scorer(metrics.r2_score),
    "rmse": make_scorer(metrics.root_mean_squared_error),
}


def _function_call_succeeds(func: Callable) -> bool:
    try:
        func()
        return True
    except AttributeError:
        return False


def _get_feature_names(estimator, X, transformer=None) -> list[str]:
    """Get the names of an estimator's input features.

    The estimator may or may not be inside a sklearn.Pipeline.
    """
    if hasattr(estimator, "feature_names_in_"):
        return estimator.feature_names_in_
    elif transformer is not None and _function_call_succeeds(
        transformer.get_feature_names_out
    ):
        # It can happen that `transformer` does have `get_feature_names_out`, but
        # calling it fails because an underlying estimator does not have that method.
        return transformer.get_feature_names_out()
    elif hasattr(X, "columns"):
        return X.columns.tolist()
    return [f"Feature #{i}" for i in range(X.shape[1])]


def _check_metric(metric: Any) -> Metric | None:
    """Check that `metric` is valid, and convert it to a suitable form as needed.

    If `metric` is a list of strings, it is checked against our own metric names.
    For example, "rmse" is recognized as root-mean-square error, even though sklearn
    itself does not recognize this name.
    Similarly, "neg_root_mean_square_error" is not recognized, and leads to an error.

    Parameters
    ----------
    metric : str, callable, list, tuple, dict, or None
        The metric to check.

    Returns
    -------
    metric
        A scoring hopefully suitable for passing to `permutation_importance`.
        Can be equal to the original metric.

    Raises
    ------
    TypeError
        If `metric` does not type-check.

    Examples
    --------
    >>> from sklearn.metrics import make_scorer, root_mean_squared_error
    >>> from skore._sklearn._estimator.feature_importance_accessor import _check_metric

    >>> _check_metric(None)  # Returns None

    >>> _check_metric(make_scorer(root_mean_squared_error))
    make_scorer(root_mean_squared_error, ...)

    >>> _check_metric({"rmse": make_scorer(root_mean_squared_error)})
    {'rmse': make_scorer(root_mean_squared_error, ...)}

    >>> _check_metric("rmse")
    {'rmse': make_scorer(root_mean_squared_error, ...)}

    >>> _check_metric(["r2", "rmse"])
    {'r2': make_scorer(r2_score, ...),
    'rmse': make_scorer(root_mean_squared_error, ...)}

    >>> _check_metric("neg_root_mean_squared_error")
    Traceback (most recent call last):
    TypeError: If metric is a string, it must be one of ...;
    got 'neg_root_mean_squared_error'

    >>> _check_metric(["r2", make_scorer(root_mean_squared_error)])
    Traceback (most recent call last):
    TypeError: If metric is a list or tuple, it must contain only strings; ...

    >>> _check_metric(3)
    Traceback (most recent call last):
    TypeError: metric must be a string, callable, list, tuple or dict;
    got <class 'int'>
    """
    if metric is None or callable(metric) or isinstance(metric, dict):
        return metric
    elif isinstance(metric, str):
        if metric in metric_to_scorer:
            # Convert to scorer
            return {metric: metric_to_scorer[cast(MetricNames, metric)]}
        raise TypeError(
            "If metric is a string, it must be one of "
            f"{list(metric_to_scorer.keys())}; got '{metric}'"
        )
    elif isinstance(metric, list | tuple):
        result: dict[str, Callable] = {}
        for s in metric:
            if isinstance(s, str):
                result |= cast(dict[str, Callable], _check_metric(s))
            else:
                raise TypeError(
                    "If metric is a list or tuple, it must contain only strings; "
                    f"got {s} of type {type(s)}"
                )
        return result
    else:
        raise TypeError(
            "metric must be a string, callable, list, tuple or dict; "
            f"got {type(metric)}"
        )


class _FeatureImportanceAccessor(_BaseAccessor[EstimatorReport], DirNamesMixin):
    """Accessor for feature importance related operations.

    You can access this accessor using the `feature_importance` attribute.
    """

    def __init__(self, parent: EstimatorReport) -> None:
        super().__init__(parent)

    @available_if(_check_estimator_has_coef())
    def coefficients(self) -> FeatureImportanceCoefficientsDisplay:
        """Retrieve the coefficients of a linear model, including the intercept.

        Returns
        -------
        :class:`FeatureImportanceCoefficientsDisplay`
            The feature importance display containing model coefficients and
            intercept.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> regressor = Ridge()
        >>> report = EstimatorReport(regressor, **split_data)
        >>> display = report.feature_importance.coefficients()
        >>> display.frame()
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
        >>> display.plot() # shows plot
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
        try:
            intercept = np.atleast_2d(estimator.intercept_)
        except AttributeError:
            # TransformedTargetRegressor() does not expose `intercept_`
            intercept = np.atleast_2d(estimator.regressor_.intercept_)
            # Uncomment when SGDOneClassSVM is fully supported by EstimatorReport
            # except AttributeError:
            # SGDOneClassSVM does not expose `intercept_`
            # intercept = None

        try:
            coef = np.atleast_2d(estimator.coef_)
        except AttributeError:
            # TransformedTargetRegressor() does not expose `coef_`
            coef = np.atleast_2d(estimator.regressor_.coef_)

        if intercept is None:
            data = coef.T
            index = list(feature_names)
        else:
            data = np.concatenate([intercept, coef.T])
            index = ["Intercept"] + list(feature_names)

        if data.shape[1] == 1:
            columns = ["Coefficient"]
        elif is_classifier(parent_estimator):
            columns = [f"Class #{i}" for i in range(data.shape[1])]
        else:
            columns = [f"Target #{i}" for i in range(data.shape[1])]

        df = pd.DataFrame(
            data=data,
            index=index,
            columns=columns,
        )

        return FeatureImportanceCoefficientsDisplay("estimator", df)

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
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = make_classification(n_features=5, random_state=42)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> forest = RandomForestClassifier(n_estimators=5, random_state=0)
        >>> report = EstimatorReport(forest, **split_data)
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

    def permutation(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        aggregate: Aggregate | None = None,
        metric: Metric | None = None,
        n_repeats: int = 5,
        max_samples: float = 1.0,
        n_jobs: int | None = None,
        seed: int | None = None,
        flat_index: bool = False,
        at_step: int | str = 0,
    ) -> pd.DataFrame:
        """Report the permutation feature importance.

        This computes the permutation importance using sklearn's
        :func:`~sklearn.inspection.permutation_importance` function,
        which consists in permuting the values of one feature and comparing
        the value of `metric` between with and without the permutation, which gives an
        indication on the impact of the feature.

        By default, `seed` is set to `None`, which means the function will
        return a different result at every call. In that case, the results are not
        cached. If you wish to take advantage of skore's caching capabilities, make
        sure you set the `seed` parameter.

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

        metric : str, callable, list, tuple, or dict, default=None
            The metric to pass to :func:`~sklearn.inspection.permutation_importance`.

            If `metric` represents a single metric, one can use:

            - a single string, which must be one of the supported metrics;
            - a callable that returns a single value.

            If `metric` represents multiple metrics, one can use:

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

        seed : int or None, default=None
            The seed used to initialize the random number generator used for the
            permutations.

        flat_index : bool, default=False
            Whether to flatten the multi-index columns. Flat index will always be lower
            case, do not include spaces and remove the hash symbol to ease indexing.

        at_step : int or str, default=0
            If the estimator is a :class:`~sklearn.pipeline.Pipeline`, at which step of
            the pipeline the importance is computed. If `n`, then the features that
            are evaluated are the ones *right before* the `n`-th step of the pipeline.
            For instance,

            - If 0, compute the importance just before the start of the pipeline (i.e.
              the importance of the raw input features).
            - If -1, compute the importance just before the end of the pipeline (i.e.
              the importance of the fully engineered features, just before the actual
              prediction step).

            If a string, will be searched among the pipeline's `named_steps`.

            Has no effect if the estimator is not a :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        pandas.DataFrame
            The permutation importance.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import Ridge
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = make_regression(n_features=3, random_state=0)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> regressor = Ridge()
        >>> report = EstimatorReport(regressor, **split_data)

        >>> report.feature_importance.permutation(
        ...    n_repeats=2,
        ...    seed=0,
        ... )
        Repeat              Repeat #0  Repeat #1
        Metric  Feature
        r2      Feature #0   0.699...   0.885...
                Feature #1   2.320...   2.636...
                Feature #2   0.028...   0.022...

        >>> report.feature_importance.permutation(
        ...    metric=["r2", "rmse"],
        ...    n_repeats=2,
        ...    seed=0,
        ... )
        Repeat             Repeat #0  Repeat #1
        Metric Feature
        r2     Feature #0   0.699...   0.885...
               Feature #1   2.320...   2.636...
               Feature #2   0.028...   0.022...
        rmse   Feature #0 -47.222... -53.231...
               Feature #1 -86.608... -92.366...
               Feature #2  -8.930...  -7.916...

        >>> report.feature_importance.permutation(
        ...    n_repeats=2,
        ...    aggregate=["mean", "std"],
        ...    seed=0,
        ... )
                                mean       std
        Metric  Feature
        r2      Feature #0  0.792...  0.131...
                Feature #1  2.478...  0.223...
                Feature #2  0.025...  0.003...

        >>> report.feature_importance.permutation(
        ...    n_repeats=2,
        ...    aggregate=["mean", "std"],
        ...    flat_index=True,
        ...    seed=0,
        ... )
                          mean       std
        r2_feature_0  0.792...  0.131...
        r2_feature_1  2.478...  0.223...
        r2_feature_2  0.025...  0.003...

        >>> # Compute the importance at the end of feature engineering pipeline
        >>> from sklearn.pipeline import make_pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> pipeline = make_pipeline(StandardScaler(), Ridge())
        >>> pipeline_report = EstimatorReport(pipeline, **split_data)
        >>> pipeline_report.feature_importance.permutation(
        ...    n_repeats=2,
        ...    seed=0,
        ...    at_step=-1,
        ... )
        Repeat         Repeat #0  Repeat #1
        Metric Feature
        r2     x0       0.699...   0.884...
               x1       2.318...   2.633...
               x2       0.028...   0.022...

        >>> pipeline_report.feature_importance.permutation(
        ...    n_repeats=2,
        ...    seed=0,
        ...    at_step="ridge",
        ... )
        Repeat         Repeat #0  Repeat #1
        Metric Feature
        r2     x0       0.699...   0.884...
               x1       2.318...   2.633...
               x2       0.028...   0.022...

        Notes
        -----
        Even if pipeline components output sparse arrays, these will be made dense.
        """
        return self._feature_permutation(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
            aggregate=aggregate,
            metric=metric,
            n_repeats=n_repeats,
            max_samples=max_samples,
            n_jobs=n_jobs,
            seed=seed,
            flat_index=flat_index,
            at_step=at_step,
        )

    def _feature_permutation(
        self,
        *,
        data_source: DataSource,
        data_source_hash: int | None,
        X: ArrayLike | None,
        y: ArrayLike | None,
        aggregate: Aggregate | None,
        metric: Metric | None,
        n_repeats: int,
        max_samples: float,
        n_jobs: int | None,
        seed: int | None,
        flat_index: bool,
        at_step: int | str,
    ) -> pd.DataFrame:
        """Private interface of `feature_permutation` to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around, or `None` and thus trigger its computation
        in the underlying process.
        """
        checked_metric = _check_metric(metric)

        if data_source_hash is None:
            X_, y_true, data_source_hash = self._get_X_y_and_data_source_hash(
                data_source=data_source, X=X, y=y
            )

        # NOTE: to temporary improve the `project.put` UX, we always store the
        # permutation importance into the cache dictionary even when seed is None.
        # Be aware that if seed is None, we still trigger the computation for all cases.
        # We only store it such that when we serialize to send to the hub, we only
        # fetch for the cache store instead of recomputing it because it is expensive.
        # FIXME: the workaround above should be removed once we are able to trigger
        # computation on the server side of skore-hub.

        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"seed must be an integer or None; got {type(seed)}")

        # build the cache key components to finally create a tuple that will be used
        # to check if the metric has already been computed
        cache_key_parts: list[Any] = [
            self._parent._hash,
            "permutation_importance",
            data_source,
            at_step,
        ]
        cache_key_parts.append(data_source_hash)

        if callable(metric) or isinstance(metric, list | dict):
            cache_key_parts.append(joblib.hash(metric))
        else:
            cache_key_parts.append(metric)

        # aggregate is not included in the cache in order to trade off computation for
        # storage
        # order arguments by key to ensure cache works n_jobs variable should not be in
        # the cache
        kwargs = {"n_repeats": n_repeats, "max_samples": max_samples, "seed": seed}
        for _, v in sorted(kwargs.items()):
            cache_key_parts.append(v)

        cache_key = tuple(cache_key_parts)

        if cache_key in self._parent._cache and seed is not None:
            # NOTE: avoid to fetch from the cache if the seed is None because we want
            # to trigger the computation in this case. We only have the permutation
            # stored as a workaround for the serialization for skore-hub as explained
            # earlier.
            score = self._parent._cache[cache_key]
        else:
            if not isinstance(self._parent.estimator_, Pipeline) or at_step == 0:
                feature_engineering, estimator = None, self._parent.estimator_
                X_transformed = X_

            else:
                pipeline = self._parent.estimator_
                if not isinstance(at_step, str | int):
                    raise ValueError(
                        f"at_step must be an integer or a string; got {at_step!r}"
                    )

                if isinstance(at_step, str):
                    # Make at_step an int and process it as usual
                    at_step = list(pipeline.named_steps.keys()).index(at_step)

                if isinstance(at_step, int):
                    if abs(at_step) >= len(pipeline.steps):
                        raise ValueError(
                            "at_step must be strictly smaller in magnitude than the "
                            "number of steps in the Pipeline, which is "
                            f"{len(pipeline.steps)}; got {at_step}"
                        )
                    feature_engineering, estimator = (
                        pipeline[:at_step],
                        pipeline[at_step:],
                    )
                    X_transformed = feature_engineering.transform(X_)

            feature_names = _get_feature_names(
                estimator,
                X=X_transformed,
                transformer=feature_engineering,
            )

            if issparse(X_transformed):
                X_transformed = X_transformed.todense()

            sklearn_score = permutation_importance(
                estimator=estimator,
                X=X_transformed,
                y=y_true,
                scoring=checked_metric,
                n_repeats=n_repeats,
                n_jobs=n_jobs,
                random_state=seed,
                max_samples=max_samples,
            )
            score = sklearn_score.get("importances")

            # If there is more than one metric
            if score is None:
                data = np.concatenate(
                    [v["importances"] for v in sklearn_score.values()]
                )
                index = pd.MultiIndex.from_product(
                    [sklearn_score, feature_names], names=("Metric", "Feature")
                )
            else:
                data = score

                # Get metric name
                if metric is None:
                    if is_classifier(estimator):
                        metric_name = "accuracy"
                    elif is_regressor(estimator):
                        metric_name = "r2"
                else:
                    # e.g. if metric is a callable
                    metric_name = None

                    # no other cases to deal with explicitly, because
                    # metric cannot possibly be a string at this stage

                if metric_name is None:
                    index = pd.Index(feature_names, name="Feature")
                else:
                    index = pd.MultiIndex.from_product(
                        [[metric_name], feature_names], names=("Metric", "Feature")
                    )

            n_repeats = data.shape[1]
            columns = pd.Index(
                (f"Repeat #{i}" for i in range(n_repeats)), name="Repeat"
            )
            score = pd.DataFrame(data=data, index=index, columns=columns)

            if cache_key is not None:
                # NOTE: for the moment, we will always store the permutation importance
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
        return self._rich_repr(class_name="skore.EstimatorReport.feature_importance")
