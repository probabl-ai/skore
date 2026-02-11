from __future__ import annotations

from collections.abc import Callable
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.sparse import issparse
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import _num_features

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._estimator.report import EstimatorReport
from skore._sklearn._plot.inspection.coefficients import CoefficientsDisplay
from skore._sklearn._plot.inspection.impurity_decrease import ImpurityDecreaseDisplay
from skore._sklearn._plot.inspection.permutation_importance import (
    PermutationImportanceDisplay,
)
from skore._sklearn.feature_names import _get_feature_names
from skore._sklearn.types import DataSource
from skore._utils._accessor import (
    _check_estimator_has_coef,
    _check_estimator_has_feature_importances,
)

Metric = str | Callable | list[str] | tuple[str] | dict[str, Callable] | None


class _InspectionAccessor(_BaseAccessor[EstimatorReport], DirNamesMixin):
    """Accessor for model inspection related operations.

    You can access this accessor using the `inspection` attribute.
    """

    _verbose_name: str = "feature_importance"

    def __init__(self, parent: EstimatorReport) -> None:
        super().__init__(parent)

    @available_if(_check_estimator_has_coef())
    def coefficients(self) -> CoefficientsDisplay:
        """Retrieve the coefficients of a linear model, including the intercept.

        Returns
        -------
        :class:`CoefficientsDisplay`
            The feature importance display containing model coefficients and
            intercept.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, shuffle=False, as_dict=True)
        >>> regressor = Ridge()
        >>> report = EstimatorReport(regressor, **split_data)
        >>> display = report.inspection.coefficients()
        >>> display.frame()
               feature  coefficient
        0    Intercept      151.4...
        1   Feature #0       30.6...
        2   Feature #1      -69.8...
        3   Feature #2      254.8...
        4   Feature #3      168.3...
        5   Feature #4       18.3...
        6   Feature #5      -19.5...
        7   Feature #6     -134.6...
        8   Feature #7      117.2...
        9   Feature #8      242.1...
        10  Feature #9      113.2...
        >>> display.plot() # shows plot
        """
        return CoefficientsDisplay._compute_data_for_display(
            estimators=[self._parent.estimator_],
            names=[self._parent.estimator_name_],
            splits=[np.nan],
            report_type="estimator",
        )

    @available_if(_check_estimator_has_feature_importances())
    def impurity_decrease(self) -> ImpurityDecreaseDisplay:
        """Retrieve the Mean Decrease in Impurity (MDI) of a tree-based model.

        This method is available for estimators that expose a `feature_importances_`
        attribute. See for example
        :attr:`sklearn.ensemble.GradientBoostingClassifier.inspections_`.
        In particular, note that the MDI is computed at fit time, i.e. using the
        training data.

        Returns
        -------
        :class:`ImpurityDecreaseDisplay`
            The feature importance display containing the mean decrease in impurity.

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
        >>> display = report.inspection.impurity_decrease()
        >>> display.frame()
              feature  importance
        0  Feature #0     0.06...
        1  Feature #1     0.19...
        2  Feature #2     0.01...
        3  Feature #3     0.69...
        4  Feature #4     0.02...
        """
        return ImpurityDecreaseDisplay._compute_data_for_display(
            estimators=[self._parent.estimator_],
            names=[self._parent.estimator_name_],
            splits=[np.nan],
            report_type="estimator",
        )

    def permutation_importance(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        at_step: int | str = 0,
        metric: Metric = None,
        n_repeats: int = 5,
        max_samples: float = 1.0,
        n_jobs: int | None = None,
        seed: int | None = None,
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

        >>> report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ... ).frame(aggregate=None)
          data_source metric     feature  repetition     value
        0        test     r2  Feature #0           1  0.69...
        1        test     r2  Feature #1           1  2.32...
        2        test     r2  Feature #2           1  0.02...
        3        test     r2  Feature #0           2  0.88...
        4        test     r2  Feature #1           2  2.63...
        5        test     r2  Feature #2           2  0.02...

        >>> report.inspection.permutation_importance(
        ...    metric=["r2", "neg_mean_squared_error"],
        ...    n_repeats=2,
        ...    seed=0,
        ... ).frame(aggregate=None)
          data_source                  metric     feature  repetition        value
        0        test                      r2  Feature #0           1     0.69...
        1        test                      r2  Feature #1           1     2.32...
        2        test                      r2  Feature #2           1     0.02...
        3        test                      r2  Feature #0           2     0.88...
        4        test                      r2  Feature #1           2     2.63...
        5        test                      r2  Feature #2           2     0.02...
        0        test  neg_mean_squared_error  Feature #0           1  2298.79...
        1        test  neg_mean_squared_error  Feature #1           1  7627.28...
        2        test  neg_mean_squared_error  Feature #2           1    92.78...
        3        test  neg_mean_squared_error  Feature #0           2  2911.23...
        4        test  neg_mean_squared_error  Feature #1           2  8666.16...
        5        test  neg_mean_squared_error  Feature #2           2    74.21...

        >>> report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ... ).frame()
          data_source metric     feature  value_mean  value_std
        0        test     r2  Feature #0    0.79...   0.13...
        1        test     r2  Feature #1    2.47...   0.22...
        2        test     r2  Feature #2    0.02...   0.00...

        >>> # Compute the importance at the end of feature engineering pipeline
        >>> from sklearn.pipeline import make_pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> pipeline = make_pipeline(StandardScaler(), Ridge())
        >>> pipeline_report = EstimatorReport(pipeline, **split_data)
        >>> pipeline_report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ...    at_step=-1,
        ... ).frame()
          data_source metric feature  value_mean  value_std
        0        test     r2      x0    0.79...   0.13...
        1        test     r2      x1    2.47...   0.22...
        2        test     r2      x2    0.02...   0.00...

        >>> pipeline_report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ...    at_step="ridge",
        ... ).frame()
          data_source metric feature  value_mean  value_std
        0        test     r2      x0    0.79...   0.13...
        1        test     r2      x1    2.47...   0.22...
        2        test     r2      x2    0.02...   0.00...

        Notes
        -----
        Even if pipeline components output sparse arrays, these will be made dense.
        """
        X_, y_true, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )
        if y_true is None:
            raise ValueError(
                "The target should be provided when computing the permutation "
                "importance."
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
            display = self._parent._cache[cache_key]
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
                n_features=_num_features(X_transformed),
                X=X_transformed,
                transformer=feature_engineering,
            )

            if issparse(X_transformed):
                X_transformed = X_transformed.todense()

            display = PermutationImportanceDisplay._compute_data_for_display(
                data_source=data_source,
                estimator=estimator,
                estimator_name=self._parent.estimator_name_,
                X=X_transformed,
                y=y_true,
                feature_names=feature_names,
                metric=metric,
                n_repeats=n_repeats,
                max_samples=max_samples,
                n_jobs=n_jobs,
                seed=seed,
                report_type="estimator",
            )

            if cache_key is not None:
                # NOTE: for the moment, we will always store the permutation importance
                self._parent._cache[cache_key] = display

        return display

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(class_name="skore.EstimatorReport.inspection")
