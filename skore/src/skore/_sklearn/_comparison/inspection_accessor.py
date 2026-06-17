from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._plot.inspection.coefficients import CoefficientsDisplay
from skore._sklearn._plot.inspection.impurity_decrease import ImpurityDecreaseDisplay
from skore._sklearn._plot.inspection.permutation_importance import (
    PermutationImportanceDisplay,
)
from skore._sklearn.metrics import MetricLike
from skore._sklearn.types import DataSource
from skore._utils._accessor import (
    _check_comparison_report_sub_estimators_have_coef,
    _check_comparison_report_sub_estimators_have_feature_importances,
)

if TYPE_CHECKING:
    from skore import ComparisonReport


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

        See Also
        --------
        :class:`CoefficientsDisplay` : Display class for coefficient plots.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import compare, evaluate
        >>> X, y = load_diabetes(return_X_y=True)
        >>> report_big_alpha = evaluate(Ridge(alpha=1e3), X, y, splitter=0.2)
        >>> report_small_alpha = evaluate(Ridge(alpha=1e-3), X, y, splitter=0.2)
        >>> report = compare({
        ...     "report small alpha": report_small_alpha,
        ...     "report big alpha": report_big_alpha,
        ... })
        >>> display = report.inspection.coefficients()
        >>> display.frame()
                     estimator     feature  coefficient
        0   report small alpha   Intercept      152.5...
        ...
        >>> display.plot() # shows plot
        """
        return CoefficientsDisplay(
            coefficients=pd.concat(
                [
                    report.inspection.coefficients()
                    .coefficients.copy()
                    .assign(estimator=name)
                    for name, report in self._parent.reports_.items()
                ],
                ignore_index=True,
            ),
            report_type=self._parent._report_type,
        )

    @available_if(_check_comparison_report_sub_estimators_have_feature_importances())
    def impurity_decrease(self) -> ImpurityDecreaseDisplay:
        """Retrieve the Mean Decrease in Impurity (MDI) for each report.

        This method is available for estimators that expose a `feature_importances_`
        attribute. See for example
        :attr:`sklearn.ensemble.GradientBoostingClassifier.feature_importances_`.

        In particular, note that the MDI is computed at fit time, i.e. using the
        training data.

        Comparison reports with the same features are put under one key and are plotted
        together. When some reports share the same features and others do not, those
        with the same features are plotted together.

        Returns
        -------
        :class:`ImpurityDecreaseDisplay`
            The impurity decrease display containing the feature importances.

        See Also
        --------
        :class:`ImpurityDecreaseDisplay` : Display class for impurity decrease plots.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from skore import compare, evaluate
        >>> X, y = load_iris(return_X_y=True, as_frame=True)
        >>> report_small_trees = evaluate(
        ...     RandomForestClassifier(max_depth=2, random_state=0), X, y, splitter=0.2
        ... )
        >>> report_big_trees = evaluate(
        ...     RandomForestClassifier(random_state=0), X, y, splitter=0.2
        ... )
        >>> report = compare({
        ...     "small trees": report_small_trees,
        ...     "big trees": report_big_trees,
        ... })
        >>> display = report.inspection.impurity_decrease()
        >>> display.frame()
             estimator            feature   importance
        0  small trees  sepal length (cm)       0.1...
        1  small trees   sepal width (cm)       0.0...
        2  small trees  petal length (cm)       0.4...
        3  small trees   petal width (cm)       0.4...
        4    big trees  sepal length (cm)       0.1...
        5    big trees   sepal width (cm)       0.0...
        6    big trees  petal length (cm)       0.4...
        7    big trees   petal width (cm)       0.4...
        >>> display.plot() # shows plot
        """
        return ImpurityDecreaseDisplay(
            importances=pd.concat(
                [
                    report.inspection.impurity_decrease()
                    .importances.copy()
                    .assign(estimator=name)
                    for name, report in self._parent.reports_.items()
                ],
                ignore_index=True,
            ),
            report_type=self._parent._report_type,
        )

    def permutation_importance(
        self,
        *,
        data_source: DataSource = "test",
        at_step: int | str = 0,
        metric: MetricLike | list[MetricLike] | dict[str, MetricLike] | None = None,
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
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

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

        metric : str, callable, scorer, or list of such instances or dict of such \
                instances, default=None
            The metric to pass to :func:`sklearn.inspection.permutation_importance`.

            - if ``None``, a suitable default will be used.
            - if a string, must be a scikit-learn scorer name. You can get the list of
              available scorers with :func:`sklearn.metrics.get_scorer_names`.
            - if a callable, must be a function with signature
              ``scorer(estimator, X, y)``.

            For more details on the accepted types, see the `scoring` argument of
            :func:`sklearn.inspection.permutation_importance`.

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

        See Also
        --------
        :class:`PermutationImportanceDisplay`
            Display class for permutation importance plots.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.metrics import make_scorer, r2_score, mean_squared_error
        >>> from skore import compare, evaluate
        >>> X, y = make_regression(n_features=3, random_state=0)
        >>> report_big_alpha = evaluate(Ridge(alpha=1e3), X, y, splitter=0.2)
        >>> report_small_alpha = evaluate(Ridge(alpha=1e-3), X, y, splitter=0.2)
        >>> report = compare({
        ...     "small alpha": report_small_alpha,
        ...     "big alpha": report_big_alpha,
        ... })
        >>> report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ... ).frame(aggregate=None)
             estimator data_source metric     feature  repetition     value
        0  small alpha        test     r2  Feature #0           1  0.77...
        1  small alpha        test     r2  Feature #1           1  1.41...
        2  small alpha        test     r2  Feature #2           1  0.01...
        ...
        >>> report.inspection.permutation_importance(
        ...    metric={
        ...        "r2": make_scorer(r2_score),
        ...        "mse": make_scorer(mean_squared_error),
        ...    },
        ...    n_repeats=2,
        ...    seed=0,
        ... ).frame()
              estimator data_source  metric  ...   value_mean    value_std
        0   small alpha        test      r2  ...     0.792711     0.029133
        1   small alpha        test      r2  ...     1.582839     0.238285
        2   small alpha        test      r2  ...     0.023727     0.005348
        3   small alpha        test     mse  ...  -2663.409431   97.882726
        4   small alpha        test     mse  ...  -5318.136563  800.606376
        5   small alpha        test     mse  ...   -79.718389    17.967898
        ...

        Notes
        -----
        Even if pipeline components output sparse arrays, these will be made dense.
        """
        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"seed must be an integer or None; got {type(seed)}")

        return PermutationImportanceDisplay(
            importances=pd.concat(
                [
                    report.inspection.permutation_importance(
                        data_source=data_source,
                        at_step=at_step,
                        metric=metric,
                        n_repeats=n_repeats,
                        max_samples=max_samples,
                        n_jobs=n_jobs,
                        seed=seed,
                    ).importances.assign(estimator=name)
                    for name, report in self._parent.reports_.items()
                ],
                ignore_index=True,
            ),
            report_type=self._parent._report_type,
        )
