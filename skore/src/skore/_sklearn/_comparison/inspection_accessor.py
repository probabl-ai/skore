from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._plot.inspection.coefficients import CoefficientsDisplay
from skore._sklearn._plot.inspection.impurity_decrease import ImpurityDecreaseDisplay
from skore._sklearn._plot.inspection.permutation_importance import (
    PermutationImportanceDisplay,
)
from skore._sklearn.types import DataSource
from skore._utils._accessor import (
    _check_comparison_report_sub_estimators_have_coef,
    _check_comparison_report_sub_estimators_have_feature_importances,
)
from skore._utils._cache_key import deep_key_sanitize

if TYPE_CHECKING:
    from skore import ComparisonReport
    from skore._sklearn._cross_validation.report import CrossValidationReport

Metric = str | Callable | list[str] | tuple[str] | dict[str, Callable] | None


class _InspectionAccessor(_BaseAccessor["ComparisonReport"], DirNamesMixin):
    """Accessor for model inspection related operations.

    You can access this accessor using the `inspection` attribute.
    """

    def __init__(self, parent: ComparisonReport) -> None:
        super().__init__(parent)

    @available_if(_check_comparison_report_sub_estimators_have_coef())
    def coefficients(self) -> CoefficientsDisplay:
        """Retrieve the coefficients for each report, including the intercepts.

        Comparison reports with the same features are put under one key and are plotted
        together. When some reports share the same features and others do not, those
        with the same features are plotted together.

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
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, shuffle=False, as_dict=True)
        >>> report_big_alpha = EstimatorReport(Ridge(alpha=1e3), **split_data)
        >>> report_small_alpha = EstimatorReport(Ridge(alpha=1e-3), **split_data)
        >>> report = ComparisonReport({
        ...     "report small alpha": report_small_alpha,
        ...     "report big alpha": report_big_alpha,
        ... })
        >>> display = report.inspection.coefficients()
        >>> display.frame()
                     estimator     feature  coefficient
        0   report small alpha   Intercept      151.5...
        1   report small alpha  Feature #0      -11.6...
        2   report small alpha  Feature #1     -238.2...
        3   report small alpha  Feature #2      505.5...
        4   report small alpha  Feature #3      298.4...
        5   report small alpha  Feature #4     -408.5...
        6   report small alpha  Feature #5      164.0...
        7   report small alpha  Feature #6      -55.1...
        8   report small alpha  Feature #7      123.0...
        9   report small alpha  Feature #8      633.8...
        10  report small alpha  Feature #9       99.4...
        11    report big alpha   Intercept      151.0...
        12    report big alpha  Feature #0        0.2...
        13    report big alpha  Feature #1        0.0...
        14    report big alpha  Feature #2        0.6...
        15    report big alpha  Feature #3        0.5...
        16    report big alpha  Feature #4        0.2...
        17    report big alpha  Feature #5        0.2...
        18    report big alpha  Feature #6       -0.4...
        19    report big alpha  Feature #7        0.5...
        20    report big alpha  Feature #8        0.6...
        21    report big alpha  Feature #9        0.4...
        >>> display.plot() # shows plot
        """
        if self._parent._report_type == "comparison-estimator":
            return CoefficientsDisplay._compute_data_for_display(
                estimators=[
                    report.estimator_ for report in self._parent.reports_.values()
                ],
                names=list(self._parent.reports_.keys()),
                splits=[np.nan] * len(self._parent.reports_),
                report_type=self._parent._report_type,
            )
        else:  # self._parent._report_type == "comparison-cross-validation"
            estimators, names = [], []
            splits: list[int | float] = []
            for name, report in self._parent.reports_.items():
                cross_validation_report = cast("CrossValidationReport", report)
                for split_idx, estimator_report in enumerate(
                    cross_validation_report.estimator_reports_
                ):
                    estimators.append(estimator_report.estimator_)
                    names.append(name)
                    splits.append(split_idx)
            return CoefficientsDisplay._compute_data_for_display(
                estimators=estimators,
                names=names,
                splits=splits,
                report_type=self._parent._report_type,
            )

    @available_if(_check_comparison_report_sub_estimators_have_feature_importances())
    def impurity_decrease(self) -> ImpurityDecreaseDisplay:
        """Retrieve the Mean Decrease in Impurity (MDI) for each report.

        This method is available for estimators that expose a `feature_importances_`
        attribute. See for example
        :attr:`sklearn.ensemble.GradientBoostingClassifier.inspections_`.

        In particular, note that the MDI is computed at fit time, i.e. using the
        training data.

        Comparison reports with the same features are put under one key and are plotted
        together. When some reports share the same features and others do not, those
        with the same features are plotted together.

        Returns
        -------
        :class:`ImpurityDecreaseDisplay`
            The impurity decrease display containing the feature importances.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_iris(return_X_y=True, as_frame=True)
        >>> split_data = train_test_split(X=X, y=y, shuffle=False, as_dict=True)
        >>> report_small_trees = EstimatorReport(
        ...     RandomForestClassifier(max_depth=2, random_state=0), **split_data
        ... )
        >>> report_big_trees = EstimatorReport(
        ...     RandomForestClassifier(random_state=0), **split_data
        ... )
        >>> report = ComparisonReport({
        ...     "small trees": report_small_trees,
        ...     "big trees": report_big_trees,
        ... })
        >>> display = report.inspection.impurity_decrease()
        >>> display.frame()
             estimator            feature   importance
        0  small trees  sepal length (cm)       0.1...
        1  small trees   sepal width (cm)      0.0...
        2  small trees  petal length (cm)       0.4...
        3  small trees   petal width (cm)       0.4...
        4    big trees  sepal length (cm)       0.0...
        5    big trees   sepal width (cm)       0.0...
        6    big trees  petal length (cm)       0.4...
        7    big trees   petal width (cm)       0.4...
        >>> display.plot() # shows plot
        """
        if self._parent._report_type == "comparison-estimator":
            return ImpurityDecreaseDisplay._compute_data_for_display(
                estimators=[
                    report.estimator_ for report in self._parent.reports_.values()
                ],
                names=list(self._parent.reports_.keys()),
                splits=[np.nan] * len(self._parent.reports_),
                report_type=self._parent._report_type,
            )
        else:  # self._parent._report_type == "comparison-cross-validation":
            estimators, names = [], []
            splits: list[int | float] = []
            for name, report in self._parent.reports_.items():
                report = cast("CrossValidationReport", report)
                for split_idx, estimator_report in enumerate(report.estimator_reports_):
                    estimators.append(estimator_report.estimator_)
                    names.append(name)
                    splits.append(split_idx)
            return ImpurityDecreaseDisplay._compute_data_for_display(
                estimators=estimators,
                names=names,
                splits=splits,
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
        """Display the permutation feature importance.

        This computes the permutation importance using sklearn's
        :func:`~sklearn.inspection.permutation_importance` function,
        which consists in permuting the values of one feature and comparing
        the value of `metric` between with and without the permutation, which gives an
        indication on the impact of the feature.

        By default, `seed` is set to `None`, which means the function will
        return a different result at every call. In that case, the results are not
        cached. If you wish to take advantage of skore's caching capabilities, make
        sure you set the `seed` parameter.

        Comparison reports with the same features are put under one key and are plotted
        together. When some reports share the same features and others do not, those
        with the same features are plotted together.

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

        Notes
        -----
        Even if pipeline components output sparse arrays, these will be made dense.
        """
        # NOTE: to temporary improve the `project.put` UX, we always store the
        # permutation importance into the cache dictionary even when seed is None.
        # Be aware that if seed is None, we still trigger the computation for all cases.
        # We only store it such that when we serialize to send to the hub, we only
        # fetch for the cache store instead of recomputing it because it is expensive.
        # FIXME: the workaround above should be removed once we are able to trigger
        # computation on the server side of skore-hub.

        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"seed must be an integer or None; got {type(seed)}")

        data_source_hash: int | None = None
        if data_source == "X_y":
            _, _, data_source_hash = self._get_X_y_and_data_source_hash(
                data_source=data_source, X=X, y=y
            )

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
            estimators, names = [], []
            splits: list[int | float] = []
            Xs: list[ArrayLike] = []
            ys: list[ArrayLike] = []

            if self._parent._report_type == "comparison-estimator":
                for name, report in self._parent.reports_.items():
                    report_X, report_y, _ = (
                        report.inspection._get_X_y_and_data_source_hash(
                            data_source=data_source, X=X, y=y
                        )
                    )
                    estimators.append(report.estimator_)
                    names.append(name)
                    splits.append(np.nan)
                    Xs.append(report_X)
                    ys.append(report_y)

            else:  # self._parent._report_type == "comparison-cross-validation"
                for name, report in self._parent.reports_.items():
                    cross_validation_report = cast("CrossValidationReport", report)
                    for split_idx, estimator_report in enumerate(
                        cross_validation_report.estimator_reports_
                    ):
                        report_X, report_y, _ = (
                            estimator_report.inspection._get_X_y_and_data_source_hash(
                                data_source=data_source, X=X, y=y
                            )
                        )
                        estimators.append(estimator_report.estimator_)
                        names.append(name)
                        splits.append(split_idx)
                        Xs.append(report_X)
                        ys.append(report_y)

            display = PermutationImportanceDisplay._compute_data_for_display(
                data_source=data_source,
                estimators=estimators,
                names=names,
                splits=splits,
                Xs=Xs,
                ys=ys,
                at_step=at_step,
                metric=metric,
                n_repeats=n_repeats,
                max_samples=max_samples,
                n_jobs=n_jobs,
                seed=seed,
                report_type=self._parent._report_type,
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
