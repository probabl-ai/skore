from __future__ import annotations

from collections.abc import Callable

from numpy.typing import ArrayLike
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot.inspection.coefficients import CoefficientsDisplay
from skore._sklearn._plot.inspection.impurity_decrease import ImpurityDecreaseDisplay
from skore._sklearn._plot.inspection.permutation_importance import (
    PermutationImportanceDisplay,
)
from skore._sklearn.types import DataSource
from skore._utils._accessor import (
    _check_cross_validation_sub_estimator_has_coef,
    _check_cross_validation_sub_estimator_has_feature_importances,
)
from skore._utils._cache_key import deep_key_sanitize

Metric = str | Callable | list[str] | tuple[str] | dict[str, Callable] | None


class _InspectionAccessor(_BaseAccessor[CrossValidationReport], DirNamesMixin):
    """Accessor for model inspection related operations.

    You can access this accessor using the `inspection` attribute.
    """

    def __init__(self, parent: CrossValidationReport) -> None:
        super().__init__(parent)

    @available_if(_check_cross_validation_sub_estimator_has_coef())
    def coefficients(self) -> CoefficientsDisplay:
        """Retrieve the coefficients across splits, including the intercept.

        Returns
        -------
        :class:`CoefficientsDisplay`
            The feature importance display containing model coefficients and
            intercept.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import Ridge
        >>> from skore import CrossValidationReport
        >>> X, y = make_regression(n_features=3, random_state=42)
        >>> report = CrossValidationReport(estimator=Ridge(), X=X, y=y, splitter=2)
        >>> display = report.inspection.coefficients()
        >>> display.frame()
            split     feature   coefficient
        0       0   Intercept       -0.1...
        1       0  Feature #0       73.2...
        2       0  Feature #1       26.6...
        3       0  Feature #2       17.1...
        4       1   Intercept        0.2...
        5       1  Feature #0       73.8...
        6       1  Feature #1       27.4...
        7       1  Feature #2       17.1...
        >>> display.plot() # shows plot
        """
        return CoefficientsDisplay._compute_data_for_display(
            estimators=[
                report.estimator_ for report in self._parent.estimator_reports_
            ],
            names=[
                report.estimator_name_ for report in self._parent.estimator_reports_
            ],
            splits=list(range(len(self._parent.estimator_reports_))),
            report_type=self._parent._report_type,
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
    ) -> PermutationImportanceDisplay:
        """Display to inspect feature importance via feature permutation.

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
        :class:`PermutationImportanceDisplay`
            The permutation importance display.

        Examples
        --------
        # TODO: fill returned values
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import Ridge
        >>> from skore import train_test_split
        >>> from skore import CrossValidationReport
        >>> X, y = make_regression(n_features=3, random_state=0)
        >>> report = CrossValidationReport(estimator=Ridge(), X=X, y=y, splitter=2)

        >>> report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ... ).frame(aggregate=None)

        >>> report.inspection.permutation_importance(
        ...    metric=["r2", "neg_mean_squared_error"],
        ...    n_repeats=2,
        ...    seed=0,
        ... ).frame(aggregate=None)

        >>> report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ... ).frame()

        >>> # Compute the importance at the end of feature engineering pipeline
        >>> from sklearn.pipeline import make_pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> pipeline = make_pipeline(StandardScaler(), Ridge())
        >>> pipeline_report = CrossValidationReport(
        ...     estimator=pipeline, X=X, y=y, splitter=2
        ... )
        >>> pipeline_report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ...    at_step=-1,
        ... ).frame()

        >>> pipeline_report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ...    at_step="ridge",
        ... ).frame()

        Notes
        -----
        Even if pipeline components output sparse arrays, these will be made dense.
        """
        if data_source == "X_y":
            X_, y_true, data_source_hash = self._get_X_y_and_data_source_hash(
                data_source=data_source, X=X, y=y
            )
        else:
            data_source_hash = None

        # NOTE: to temporary improve the `project.put` UX, we always store the
        # permutation importance into the cache dictionary even when seed is None.
        # Be aware that if seed is None, we still trigger the computation for all cases.
        # We only store it such that when we serialize to send to the hub, we only
        # fetch for the cache store instead of recomputing it because it is expensive.
        # FIXME: the workaround above should be removed once we are able to trigger
        # computation on the server side of skore-hub.

        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"seed must be an integer or None; got {type(seed)}")

        # n_jobs should not be in cache
        kwargs = {"n_repeats": n_repeats, "max_samples": max_samples, "seed": seed}
        cache_key = deep_key_sanitize(
            (
                self._parent._hash,
                "permutation_importance",
                data_source,
                at_step,
                data_source_hash,
                metric,
                kwargs,
            )
        )

        # NOTE: avoid to fetch from the cache if the seed is None because we want
        # to trigger the computation in this case. We only have the permutation
        # stored as a workaround for the serialization for skore-hub as explained
        # earlier.
        display = None if seed is None else self._parent._cache.get(cache_key)
        if display is None:
            Xs: list[ArrayLike] = []
            ys: list[ArrayLike] = []
            for report in self._parent.estimator_reports_:
                if data_source == "X_y":
                    Xs.append(X_)
                    ys.append(y_true)
                else:
                    report_X, report_y, _ = (
                        report.inspection._get_X_y_and_data_source_hash(
                            data_source=data_source
                        )
                    )
                    if report_y is None:
                        raise ValueError(
                            "Permutation importance can not be performed on a "
                            "clustering model."
                        )
                    Xs.append(report_X)
                    ys.append(report_y)

            display = PermutationImportanceDisplay._compute_data_for_display(
                data_source=data_source,
                estimators=[
                    report.estimator_ for report in self._parent.estimator_reports_
                ],
                names=[
                    report.estimator_name_ for report in self._parent.estimator_reports_
                ],
                splits=list(range(len(self._parent.estimator_reports_))),
                Xs=Xs,
                ys=ys,
                at_step=at_step,
                metric=metric,
                n_repeats=n_repeats,
                max_samples=max_samples,
                n_jobs=n_jobs,
                seed=seed,
                report_type="cross-validation",
            )

            if cache_key is not None:
                # NOTE: for the moment, we will always store the permutation importance
                self._parent._cache[cache_key] = display

        return display

    @available_if(_check_cross_validation_sub_estimator_has_feature_importances())
    def impurity_decrease(self) -> ImpurityDecreaseDisplay:
        """Retrieve the Mean Decrease in Impurity (MDI) for each split.

        This method is available for estimators that expose a `feature_importances_`
        attribute. See for example
        :attr:`sklearn.ensemble.GradientBoostingClassifier.feature_importances_`.

        In particular, note that the MDI is computed at fit time, i.e. using the
        training data.

        Returns
        -------
        :class:`ImpurityDecreaseDisplay`
            The impurity decrease display containing the feature importances.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from skore import CrossValidationReport
        >>> iris = load_iris(as_frame=True)
        >>> X, y = iris.data, iris.target
        >>> y = iris.target_names[y]
        >>> report = CrossValidationReport(
        ...     estimator=RandomForestClassifier(random_state=0), X=X, y=y, splitter=5
        ... )
        >>> display = report.inspection.impurity_decrease()
        >>> display.frame()
            split            feature  importance
        0       0  sepal length (cm)       0.0...
        1       0   sepal width (cm)       0.0...
        2       0  petal length (cm)       0.4...
        3       0   petal width (cm)       0.4...
        4       1  sepal length (cm)       0.0...
        ...
        >>> display.plot() # shows plot
        """
        return ImpurityDecreaseDisplay._compute_data_for_display(
            estimators=[
                report.estimator_ for report in self._parent.estimator_reports_
            ],
            names=[
                report.estimator_name_ for report in self._parent.estimator_reports_
            ],
            splits=list(range(len(self._parent.estimator_reports_))),
            report_type=self._parent._report_type,
        )

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(class_name="skore.CrossValidationReport.inspection")
