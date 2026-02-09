from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
import seaborn
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from skore._sklearn._plot.base import BOXPLOT_STYLE, DisplayMixin
from skore._sklearn._plot.inspection.utils import _decorate_matplotlib_axis
from skore._sklearn.feature_names import _get_feature_names
from skore._sklearn.types import ReportType


class ImpurityDecreaseDisplay(DisplayMixin):
    """Display to inspect the Mean Decrease in Impurity (MDI) of tree-based models.

    Parameters
    ----------
    importances : DataFrame
        The importances data to display. The columns are:

        - `estimator`
        - `feature`
        - `importance`

    report_type : {"estimator", "cross-validation", "comparison-estimator", \
            "comparison-cross-validation"}
        Report type from which the display is created.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from skore import EstimatorReport, train_test_split
    >>> iris = load_iris(as_frame=True)
    >>> X, y = iris.data, iris.target
    >>> y = iris.target_names[y]
    >>> split_data = train_test_split(
    ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
    ... )
    >>> report = EstimatorReport(
    ...     RandomForestClassifier(random_state=0), **split_data
    ... )
    >>> display = report.inspection.impurity_decrease()
    >>> display.frame()
                    feature  importance
    0  sepal length (cm)     0.1...
    1   sepal width (cm)     0.0...
    2  petal length (cm)     0.4...
    3   petal width (cm)     0.3...
    """

    _default_barplot_kwargs: dict[str, Any] = {
        "aspect": 2,
        "height": 6,
    }
    _default_stripplot_kwargs: dict[str, Any] = {
        "alpha": 0.5,
        "aspect": 2,
        "height": 6,
    }
    _default_boxplot_kwargs: dict[str, Any] = {
        "whis": 1e10,
        **BOXPLOT_STYLE,
    }

    def __init__(self, *, importances: pd.DataFrame, report_type: ReportType):
        self.importances = importances
        self.report_type = report_type

    @classmethod
    def _compute_data_for_display(
        cls,
        *,
        estimators: Sequence[BaseEstimator],
        names: list[str],
        splits: list[int | float],
        report_type: ReportType,
    ) -> ImpurityDecreaseDisplay:
        """Compute the data for the display.

        Parameters
        ----------
        estimators : list of estimator
            The estimators to compute the data for.

        names : list of str
            The names of the estimators.

        splits : list of int or np.nan
            The splits to compute the data for.

        report_type : {"estimator", "cross-validation", "comparison-estimator", \
                "comparison-cross-validation"}
            The type of report to compute the data for.

        Returns
        -------
        ImpurityDecreaseDisplay
            The data for the display.
        """
        feature_names, est_names, importances_list, split_indices = [], [], [], []
        for estimator, name, split in zip(estimators, names, splits, strict=True):
            if isinstance(estimator, Pipeline):
                preprocessor, predictor = estimator[:-1], estimator[-1]
            else:
                preprocessor, predictor = None, estimator

            n_features = predictor.feature_importances_.shape[0]
            feat_names = _get_feature_names(
                predictor, transformer=preprocessor, n_features=n_features
            )

            feature_names.extend(feat_names)
            est_names.extend([name] * len(feat_names))
            importances_list.extend(predictor.feature_importances_.tolist())
            split_indices.extend([split] * len(feat_names))

        importances = pd.DataFrame(
            {
                "estimator": est_names,
                "split": split_indices,
                "feature": feature_names,
                "importance": importances_list,
            }
        )

        return cls(importances=importances, report_type=report_type)

    def frame(self) -> pd.DataFrame:
        """Get the mean decrease in impurity in a dataframe format.

        The returned dataframe is not going to contain constant columns or columns
        containing only NaN values.

        Returns
        -------
        DataFrame
            Dataframe containing the mean decrease in impurity of the tree-based model.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from skore import EstimatorReport, train_test_split
        >>> iris = load_iris(as_frame=True)
        >>> X, y = iris.data, iris.target
        >>> y = iris.target_names[y]
        >>> split_data = train_test_split(
        ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
        ... )
        >>> report = EstimatorReport(
        ...     RandomForestClassifier(random_state=0), **split_data
        ... )
        >>> display = report.inspection.impurity_decrease()
        >>> display.frame()
                     feature  importance
        0  sepal length (cm)     0.1...
        1   sepal width (cm)     0.0...
        2  petal length (cm)     0.4...
        3   petal width (cm)     0.3...
        """
        if self.report_type == "estimator":
            columns_to_drop = ["estimator", "split"]
        elif self.report_type == "cross-validation":
            columns_to_drop = ["estimator"]
        elif self.report_type in (
            "comparison-estimator",
            "comparison-cross-validation",
        ):
            columns_to_drop = []
        else:
            raise TypeError(f"Unexpected report type: {self.report_type!r}")

        return self.importances.drop(columns=columns_to_drop)

    @DisplayMixin.style_plot
    def plot(self) -> seaborn.FacetGrid:
        """Plot the mean decrease in impurity for the different features.

        Returns
        -------
        seaborn.FacetGrid
            The generated plot.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from skore import EstimatorReport, train_test_split
        >>> iris = load_iris(as_frame=True)
        >>> X, y = iris.data, iris.target
        >>> y = iris.target_names[y]
        >>> split_data = train_test_split(
        ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
        ... )
        >>> report = EstimatorReport(RandomForestClassifier(), **split_data)
        >>> display = report.inspection.impurity_decrease()
        >>> display.plot()
        """
        return self._plot()

    def _plot_matplotlib(self) -> seaborn.FacetGrid:
        """Dispatch the plotting function for matplotlib backend.

        This method creates a bar plot showing the mean decrease in impurity for each
        feature using seaborn's catplot. For cross-validation reports, it uses a
        strip plot with boxplot overlay to show the distribution across splits.
        """
        barplot_kwargs = self._default_barplot_kwargs.copy()
        stripplot_kwargs = self._default_stripplot_kwargs.copy()
        boxplot_kwargs = self._default_boxplot_kwargs.copy()
        frame = self.frame()

        return self._plot_single_estimator(
            frame=frame,
            estimator_name=self.importances["estimator"].unique()[0],
            report_type=self.report_type,
            barplot_kwargs=barplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
        )

    def _plot_single_estimator(
        self,
        *,
        frame: pd.DataFrame,
        estimator_name: str,
        report_type: ReportType,
        barplot_kwargs: dict[str, Any],
        stripplot_kwargs: dict[str, Any],
        boxplot_kwargs: dict[str, Any],
    ) -> seaborn.FacetGrid:
        """Plot the mean decrease in impurity.

        For EstimatorReport, a bar plot is used to display the mean decrease in impurity
        values. For CrossValidationReport, a strip plot with boxplot overlay is used to
        show the distribution across splits.

        Parameters
        ----------
        frame : pd.DataFrame
            The frame to plot.

        estimator_name : str
            The name of the estimator to plot.

        report_type : {"estimator", "cross-validation", "comparison-estimator", \
                "comparison-cross-validation"}
            The type of report to compute the data for.

        barplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.barplot` for
            rendering the mean decrease in impurity with an
            :class:`~skore.EstimatorReport`.

        stripplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.stripplot` for
            rendering the mean decrease in impurity with a
            :class:`~skore.CrossValidationReport`.

        boxplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.boxplot` for
            rendering the mean decrease in impurity with a
            :class:`~skore.CrossValidationReport`.
        """
        if "estimator" in report_type:
            facet_ = sns.catplot(
                data=frame,
                x="importance",
                y="feature",
                kind="bar",
                **barplot_kwargs,
            )
        else:  # "cross-validation" in report_type
            facet_ = sns.catplot(
                data=frame,
                x="importance",
                y="feature",
                kind="strip",
                dodge=True,
                **stripplot_kwargs,
            ).map_dataframe(
                sns.boxplot,
                x="importance",
                y="feature",
                dodge=True,
                palette="tab10",
                **boxplot_kwargs,
            )

        ax_ = facet_.axes.flatten()[0]
        _decorate_matplotlib_axis(
            ax=ax_,
            add_background_features=False,
            n_features=frame["feature"].nunique(),
            xlabel="Mean Decrease in Impurity (MDI)",
            ylabel="",
        )
        facet_.figure.suptitle(f"Mean Decrease in Impurity (MDI) of {estimator_name}")
        return facet_

    # ignore the type signature because we override kwargs by specifying the name of
    # the parameters for the user.
    def set_style(  # type: ignore[override]
        self,
        *,
        policy: Literal["override", "update"] = "update",
        barplot_kwargs: dict[str, Any] | None = None,
        stripplot_kwargs: dict[str, Any] | None = None,
        boxplot_kwargs: dict[str, Any] | None = None,
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : {"override", "update"}, default="update"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        barplot_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`seaborn.barplot` for
            rendering the mean decrease in impurity with an
            :class:`~skore.EstimatorReport`.

        stripplot_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`seaborn.stripplot` for
            rendering the mean decrease in impurity with a
            :class:`~skore.CrossValidationReport`.

        boxplot_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`seaborn.boxplot` for
            rendering the mean decrease in impurity with a
            :class:`~skore.CrossValidationReport`.

        Returns
        -------
        self : object
            The instance with a modified style.

        Raises
        ------
        ValueError
            If a style parameter is unknown.
        """
        return super().set_style(
            policy=policy,
            barplot_kwargs=barplot_kwargs or {},
            stripplot_kwargs=stripplot_kwargs or {},
            boxplot_kwargs=boxplot_kwargs or {},
        )
