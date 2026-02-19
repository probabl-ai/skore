from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
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
        - `split`
        - `feature`
        - `importance`

    report_type : {"estimator", "cross-validation", "comparison-estimator", \
            "comparison-cross-validation"}
        Report type from which the display is created.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Matplotlib Axes with the plot.

    facet_ : seaborn FacetGrid
        FacetGrid containing the plot.

    figure_ : matplotlib Figure
        Figure containing the plot.

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
        elif self.report_type == "comparison-estimator":
            columns_to_drop = ["split"]
        else:  # comparison-cross-validation
            columns_to_drop = []

        return self.importances.drop(columns=columns_to_drop)

    @DisplayMixin.style_plot
    def plot(self) -> None:
        """Plot the mean decrease in impurity for the different features.

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

    def _plot_matplotlib(self) -> None:
        """Dispatch the plotting function for matplotlib backend.

        This method creates a bar plot showing the mean decrease in impurity for each
        feature using seaborn's catplot. For cross-validation reports, it uses a
        strip plot with boxplot overlay to show the distribution across splits.
        """
        # Make copy of the dictionary since we are going to pop some keys later
        barplot_kwargs = self._default_barplot_kwargs.copy()
        boxplot_kwargs = self._default_boxplot_kwargs.copy()
        stripplot_kwargs = self._default_stripplot_kwargs.copy()

        if "comparison" in self.report_type:
            return self._plot_comparison(
                frame=self.frame(),
                report_type=self.report_type,
                barplot_kwargs=barplot_kwargs,
                boxplot_kwargs=boxplot_kwargs,
                stripplot_kwargs=stripplot_kwargs,
            )
        # EstimatorReport or CrossValidationReport
        return self._plot_single_estimator(
            frame=self.frame(),
            estimator_name=self.importances["estimator"][0],
            report_type=self.report_type,
            barplot_kwargs=barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
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
    ) -> None:
        """Plot the mean decrease in impurity.

        For `EstimatorReport`, a bar plot is used to display the mean decrease in
        impurity values. For `CrossValidationReport`, a strip plot with boxplot overlay
        is used to show the distribution across splits.

        Parameters
        ----------
        frame : pd.DataFrame
            The frame to plot.

        estimator_name : str
            The name of the estimator to plot.

        report_type : {"estimator", "cross-validation"}
            The type of report to plot.

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
        self._categorical_plot(
            frame=frame,
            report_type=report_type,
            hue=None,
            col=None,
            barplot_kwargs=barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
        )

        self.figure_.suptitle(f"Mean decrease in impurity (MDI) of {estimator_name}")

    def _categorical_plot(
        self,
        *,
        frame: pd.DataFrame,
        report_type: ReportType,
        hue: str | None = None,
        col: str | None = None,
        barplot_kwargs: dict[str, Any] | None = None,
        boxplot_kwargs: dict[str, Any] | None = None,
        stripplot_kwargs: dict[str, Any] | None = None,
    ):
        if "estimator" in report_type:
            self.facet_ = sns.catplot(
                data=frame,
                x="importance",
                y="feature",
                hue=hue,
                col=col,
                kind="bar",
                **barplot_kwargs,
            )
        else:  # "cross-validation" in report_type
            self.facet_ = sns.catplot(
                data=frame,
                x="importance",
                y="feature",
                hue=hue,
                col=col,
                kind="strip",
                dodge=True,
                **stripplot_kwargs,
            ).map_dataframe(
                sns.boxplot,
                x="importance",
                y="feature",
                hue=hue,
                palette="tab10",
                **boxplot_kwargs,
            )
        add_background_features = hue is not None

        self.figure_, self.ax_ = self.facet_.figure, self.facet_.axes.squeeze()
        n_features = (
            [frame["feature"].nunique()]
            if col is None
            else [
                frame.query(f"{col} == '{col_value}'")["feature"].nunique()
                for col_value in frame[col].unique()
            ]
        )
        for ax, n_feature in zip(self.ax_.flatten(), n_features, strict=True):
            _decorate_matplotlib_axis(
                ax=ax,
                add_background_features=add_background_features,
                n_features=n_feature,
                xlabel="Mean decrease in impurity",
                ylabel="",
            )
        if len(self.ax_.flatten()) == 1:
            self.ax_ = self.ax_.flatten()[0]

    @staticmethod
    def _has_same_features(*, frame: pd.DataFrame) -> bool:
        """Check if the features are the same across all estimators."""
        grouped = {
            name: group["feature"].sort_values().tolist()
            for name, group in frame.groupby("estimator", sort=False)
        }
        _, reference_features = grouped.popitem()
        for group_features in grouped.values():
            if group_features != reference_features:
                return False
        return True

    def _plot_comparison(
        self,
        *,
        frame: pd.DataFrame,
        report_type: ReportType,
        barplot_kwargs: dict[str, Any],
        boxplot_kwargs: dict[str, Any],
        stripplot_kwargs: dict[str, Any],
    ) -> None:
        """Plot the mean decrease in impurity for a `ComparisonReport`.

        Parameters
        ----------
        frame : pd.DataFrame
            The frame to plot.

        report_type : {"comparison-estimator", "comparison-cross-validation"}
            The type of report to plot.

        barplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.barplot` for
            rendering the mean decrease in impurity with an
            :class:`~skore.ComparisonReport` of :class:`~skore.EstimatorReport`.

        boxplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.boxplot` for
            rendering the mean decrease in impurity with a
            :class:`~skore.ComparisonReport` of :class:`~skore.CrossValidationReport`.

        stripplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.stripplot` for
            rendering the mean decrease in impurity with a
            :class:`~skore.ComparisonReport` of :class:`~skore.CrossValidationReport`.
        """
        # help mypy to understand the following variable types
        hue: str | None = None

        has_same_features = self._has_same_features(frame=frame)
        if not has_same_features:
            # features cannot be compared across estimators and we therefore
            # need to subplot by estimator
            hue, col = None, "estimator"
        else:
            hue, col = "estimator", None

        self._categorical_plot(
            frame=frame,
            report_type=report_type,
            hue=hue,
            col=col,
            barplot_kwargs={"sharey": has_same_features} | barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs={"sharey": has_same_features} | stripplot_kwargs,
        )
        self.figure_.suptitle("Mean decrease in impurity (MDI)")

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
