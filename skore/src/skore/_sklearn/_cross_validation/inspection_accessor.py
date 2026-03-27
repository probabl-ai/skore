from __future__ import annotations

import pandas as pd
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot.inspection.coefficients import CoefficientsDisplay
from skore._sklearn._plot.inspection.impurity_decrease import ImpurityDecreaseDisplay
from skore._sklearn._plot.inspection.permutation_importance import (
    PermutationImportanceDisplay,
)
from skore._sklearn.types import DataSource, Metric
from skore._utils._accessor import (
    _check_cross_validation_sub_estimator_has_coef,
    _check_cross_validation_sub_estimator_has_feature_importances,
)


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
        >>> from skore import evaluate
        >>> X, y = make_regression(n_features=3, random_state=42)
        >>> report = evaluate(Ridge(), X, y, splitter=2)
        >>> display = report.inspection.coefficients()
        >>> display.frame()
              feature  coefficient_mean  coefficient_std
        0   Intercept             0.0...            0.2...
        1  Feature #0            73.5...            0.4...
        2  Feature #1            27.0...            0.5...
        3  Feature #2            17.1...            0.0...
        >>> display.plot() # shows plot
        """
        return CoefficientsDisplay(
            coefficients=pd.concat(
                [
                    report.inspection.coefficients()
                    .coefficients.copy()
                    .assign(split=split_idx)
                    for split_idx, report in enumerate(self._parent.estimator_reports_)
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
        metric: Metric | list[Metric] | dict[str, Metric] | None = None,
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
            The metric to pass to :func:`~sklearn.inspection.permutation_importance`.
            The possible values (whether or not in a list) are:

            - if a string, either one of the built-in metrics or a scikit-learn scorer
              name. You can get the possible list of string using
              `report.metrics.help()` or :func:`sklearn.metrics.get_scorer_names` for
              the built-in metrics or the scikit-learn scorers, respectively.
            - if a callable, it should take as arguments `y_true`, `y_pred` as the two
              first arguments. Additional arguments can be passed as keyword arguments
              and will be forwarded with `metric_kwargs`. No favorability indicator can
              be displayed in this case.
            - if the callable API is too restrictive (e.g. need to pass
              same parameter name with different values), you can use scikit-learn
              scorers as provided by :func:`sklearn.metrics.make_scorer`. In this case,
              the metric favorability will only be displayed if it is given explicitly
              via `make_scorer`'s `greater_is_better` parameter.

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
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import Ridge
        >>> from skore import evaluate
        >>> X, y = make_regression(n_features=3, random_state=0)
        >>> report = evaluate(Ridge(), X, y, splitter=2)
        >>> report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ... ).frame(aggregate=None)
           data_source metric  split     feature  repetition     value
        0         test     r2      0  Feature #0           1  0.71...
        1         test     r2      0  Feature #1           1  1.59...
        2         test     r2      0  Feature #2           1  0.01...
        3         test     r2      0  Feature #0           2  0.70...
        4         test     r2      0  Feature #1           2  1.58...
        5         test     r2      0  Feature #2           2  0.01...
        6         test     r2      1  Feature #0           1  0.63...
        7         test     r2      1  Feature #1           1  1.82...
        8         test     r2      1  Feature #2           1  0.01...
        9         test     r2      1  Feature #0           2  0.49...
        10        test     r2      1  Feature #1           2  1.15...
        11        test     r2      1  Feature #2           2  0.01...
        >>> report.inspection.permutation_importance(
        ...    metric=["r2", "neg_mean_squared_error"],
        ...    n_repeats=2,
        ...    seed=0,
        ... ).frame(aggregate=None)
           data_source                  metric  ...  repetition         value
        0         test                      r2  ...           1      0.71...
        1         test                      r2  ...           1      1.59...
        2         test                      r2  ...           1      0.01...
        3         test                      r2  ...           2      0.70...
        4         test                      r2  ...           2      1.58...
        5         test                      r2  ...           2      0.01...
        ...
        >>> report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ... ).frame()
          data_source metric     feature  value_mean  value_std
        0        test     r2  Feature #0     0.63...    0.10...
        1        test     r2  Feature #1     1.54...    0.07...
        2        test     r2  Feature #2     0.01...    0.00...
        >>> report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ... ).frame(level="repetitions")
          data_source metric  split     feature  value_mean  value_std
        0        test     r2      0  Feature #0     0.71...    0.00...
        1        test     r2      0  Feature #1     1.58...    0.00...
        2        test     r2      0  Feature #2     0.01...    0.00...
        3        test     r2      1  Feature #0     0.56...    0.09...
        4        test     r2      1  Feature #1     1.49...    0.47...
        5        test     r2      1  Feature #2     0.01...    0.00...
        >>> # Compute the importance at the end of feature engineering pipeline
        >>> from sklearn.pipeline import make_pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> pipeline = make_pipeline(StandardScaler(), Ridge())
        >>> pipeline_report = evaluate(pipeline, X, y, splitter=2)
        >>> pipeline_report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ...    at_step=-1,
        ... ).frame()
          data_source metric feature  value_mean  value_std
        0        test     r2      x0    0.63...     0.10...
        1        test     r2      x1    1.53...     0.06...
        2        test     r2      x2    0.01...     0.00...
        >>> pipeline_report.inspection.permutation_importance(
        ...    n_repeats=2,
        ...    seed=0,
        ...    at_step="ridge",
        ... ).frame()
          data_source metric feature  value_mean  value_std
        0        test     r2      x0    0.63...     0.10...
        1        test     r2      x1    1.53...     0.06...
        2        test     r2      x2    0.01...     0.00...

        Notes
        -----
        Even if pipeline components output sparse arrays, these will be made dense.
        """  # noqa: E501
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
                    ).importances.assign(split=split_idx)
                    for split_idx, report in enumerate(self._parent.estimator_reports_)
                ],
                ignore_index=True,
            ),
            report_type=self._parent._report_type,
        )

    def _get_cached_permutation_importances(self, data_source):
        # NOTE: this a public developer API, breaking it might break `project.put`
        common_cache_keys = set.intersection(
            *[set(report._cache) for report in self._parent.estimator_reports_]
        )
        sub_report = self._parent.estimator_reports_[0]
        for key in common_cache_keys:
            ds, name, _ = key
            if ds != data_source or name != "permutation_importance":
                continue
            _, kwargs = sub_report._cache[key]
            yield kwargs

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
        >>> from skore import evaluate
        >>> iris = load_iris(as_frame=True)
        >>> X, y = iris.data, iris.target
        >>> y = iris.target_names[y]
        >>> report = evaluate(RandomForestClassifier(random_state=0), X, y, splitter=5)
        >>> display = report.inspection.impurity_decrease()
        >>> display.frame()
                     feature  importance_mean  importance_std
        0  sepal length (cm)            0.0...           0.0...
        1   sepal width (cm)            0.0...           0.0...
        2  petal length (cm)            0.4...           0.0...
        3   petal width (cm)            0.4...           0.0...
        ...
        >>> display.plot() # shows plot
        """
        return ImpurityDecreaseDisplay(
            importances=pd.concat(
                [
                    report.inspection.impurity_decrease()
                    .importances.copy()
                    .assign(split=split_idx)
                    for split_idx, report in enumerate(self._parent.estimator_reports_)
                ],
                ignore_index=True,
            ),
            report_type=self._parent._report_type,
        )

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(class_name="skore.CrossValidationReport.inspection")
