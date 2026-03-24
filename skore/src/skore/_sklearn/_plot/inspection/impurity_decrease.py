from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from skore._sklearn._plot.base import BOXPLOT_STYLE, DisplayMixin
from skore._sklearn._plot.inspection.utils import (
    _decorate_matplotlib_axis,
    select_k_features,
    sort_features,
)
from skore._sklearn.feature_names import _get_feature_names
from skore._sklearn.types import Aggregate, ReportType
from skore._utils._index import flatten_multi_index


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

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from skore import evaluate
    >>> iris = load_iris(as_frame=True)
    >>> X, y = iris.data, iris.target
    >>> y = iris.target_names[y]
    >>> report = evaluate(RandomForestClassifier(random_state=0), X, y, splitter=0.2)
    >>> display = report.inspection.impurity_decrease()
    >>> display.frame()
                    feature  importance
    0  sepal length (cm)     0.1...
    1   sepal width (cm)     0.0...
    2  petal length (cm)     0.4...
    3   petal width (cm)     0.4...
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
        estimator: BaseEstimator,
        name: str,
        report_type: ReportType,
    ) -> ImpurityDecreaseDisplay:
        """Compute the data for the display from a single estimator.

        Parameters
        ----------
        estimator : estimator
            The estimator to compute the data for.

        name : str
            The name of the estimator.

        report_type : {"estimator", "cross-validation", "comparison-estimator", \
                "comparison-cross-validation"}
            The type of report to compute the data for.

        Returns
        -------
        ImpurityDecreaseDisplay
            The data for the display.
        """
        if isinstance(estimator, Pipeline):
            preprocessor, predictor = estimator[:-1], estimator[-1]
        else:
            preprocessor, predictor = None, estimator

        n_features = predictor.feature_importances_.shape[0]
        feature_names = _get_feature_names(
            predictor, transformer=preprocessor, n_features=n_features
        )

        importances = pd.DataFrame(
            {
                "estimator": [name] * n_features,
                "split": [np.nan] * n_features,
                "feature": feature_names,
                "importance": predictor.feature_importances_.tolist(),
            }
        )

        return cls(importances=importances, report_type=report_type)

    def frame(
        self,
        *,
        aggregate: Aggregate | None = ("mean", "std"),
        select_k: int | None = None,
        sorting_order: Literal["descending", "ascending", None] = None,
    ) -> pd.DataFrame:
        """Get the mean decrease in impurity in a dataframe format.

        Parameters
        ----------
        aggregate : {"mean", "std"}, ("mean", "std") or None, default=("mean", "std")
            Aggregate the importances over splits. Only relevant when
            ``report_type`` is ``"cross-validation"`` or
            ``"comparison-cross-validation"``; ignored otherwise. If ``None``,
            the raw per-split values are returned.

        select_k : int, default=None
            Select features by importance: positive for top k, negative for
            bottom k. Selection is per estimator when applicable. For
            cross-validation, ranking uses mean importance across splits.
            When ``aggregate`` is ``None``, ranking uses mean importance per
            feature over splits; all split rows are kept for selected features.

        sorting_order : {"descending", "ascending", None}, default=None
            Sort features by importance (descending = most important first).
            When ``aggregate`` is ``None``, ordering uses mean importance per
            feature over splits.

        Returns
        -------
        DataFrame
            Dataframe containing the mean decrease in impurity. When
            ``aggregate`` is not ``None`` and the report type involves
            cross-validation splits, the ``split`` column is removed and
            ``importance`` is replaced by aggregated columns:
            ``importance_mean`` and ``importance_std``.
        """
        if self.report_type == "estimator":
            columns_to_drop = ["estimator", "split"]
        elif self.report_type == "cross-validation":
            columns_to_drop = ["estimator"]
        elif self.report_type == "comparison-estimator":
            columns_to_drop = ["split"]
        else:  # comparison-cross-validation
            columns_to_drop = []

        frame = self.importances.drop(columns=columns_to_drop)

        if sorting_order is not None:
            frame = sort_features(
                frame,
                sorting_order,
                group_columns=[
                    c for c in self._get_columns_to_groupby(frame=frame) if c != "split"
                ],
                importance_column="importance",
            )
        if select_k is not None:
            frame = select_k_features(
                frame,
                select_k,
                group_columns=[
                    c for c in self._get_columns_to_groupby(frame=frame) if c != "split"
                ],
                importance_column="importance",
            )

        if aggregate is not None and "split" in frame.columns:
            group_by = [c for c in ["estimator", "feature"] if c in frame.columns]
            frame = (
                frame.drop(columns=["split"])
                .groupby(group_by, sort=False, dropna=False)
                .aggregate(aggregate)
            ).reset_index()
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = flatten_multi_index(frame.columns)

        return frame

    @staticmethod
    def _get_columns_to_groupby(*, frame: pd.DataFrame) -> list[str]:
        columns_to_groupby = list[str]()
        if "estimator" in frame.columns and frame["estimator"].nunique() > 1:
            columns_to_groupby.append("estimator")
        if "split" in frame.columns and frame["split"].nunique() > 1:
            columns_to_groupby.append("split")
        return columns_to_groupby

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        select_k: int | None = None,
        sorting_order: Literal["descending", "ascending", None] = None,
    ) -> Figure:
        """Plot the mean decrease in impurity for the different features.

        Parameters
        ----------
        select_k : int, default=None
            If set, only the top (positive) or bottom (negative) k features
            by importance are shown. See :meth:`frame` for details.

        sorting_order : {"descending", "ascending", None}, default=None
            Sort features by importance before plotting. See :meth:`frame`
            for details.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the MDI plot.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from skore import evaluate
        >>> iris = load_iris(as_frame=True)
        >>> X, y = iris.data, iris.target
        >>> y = iris.target_names[y]
        >>> report = evaluate(RandomForestClassifier(), X, y, splitter=0.2)
        >>> display = report.inspection.impurity_decrease()
        >>> display.plot()
        """
        return self._plot(select_k=select_k, sorting_order=sorting_order)

    def _plot_matplotlib(
        self,
        *,
        select_k: int | None = None,
        sorting_order: Literal["descending", "ascending", None] = None,
    ) -> Figure:
        """Dispatch the plotting function for matplotlib backend.

        This method creates a bar plot showing the mean decrease in impurity for each
        feature using seaborn's catplot. For cross-validation reports, it uses a
        strip plot with boxplot overlay to show the distribution across splits.
        """
        # Make copy of the dictionary since we are going to pop some keys later
        barplot_kwargs = self._default_barplot_kwargs.copy()
        boxplot_kwargs = self._default_boxplot_kwargs.copy()
        stripplot_kwargs = self._default_stripplot_kwargs.copy()

        if select_k == 0:
            raise ValueError(
                "select_k=0 would produce an empty plot. Use a non-zero value or "
                "omit select_k to plot all features."
            )

        frame = self.frame(
            aggregate=None,
            select_k=select_k,
            sorting_order=sorting_order,
        )
        if "comparison" in self.report_type:
            return self._plot_comparison(
                frame=frame,
                report_type=self.report_type,
                barplot_kwargs=barplot_kwargs,
                boxplot_kwargs=boxplot_kwargs,
                stripplot_kwargs=stripplot_kwargs,
            )
        # EstimatorReport or CrossValidationReport
        return self._plot_single_estimator(
            frame=frame,
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
    ) -> Figure:
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
        figure = self._categorical_plot(
            frame=frame,
            report_type=report_type,
            hue=None,
            col=None,
            barplot_kwargs=barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
        )

        figure.suptitle(f"Mean decrease in impurity (MDI) of {estimator_name}")
        return figure

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
    ) -> Figure:
        if "estimator" in report_type:
            facet = sns.catplot(
                data=frame,
                x="importance",
                y="feature",
                hue=hue,
                col=col,
                kind="bar",
                **barplot_kwargs,
            )
        else:  # "cross-validation" in report_type
            facet = sns.catplot(
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

        figure = facet.figure
        ax_grid = facet.axes.squeeze()
        n_features = (
            [frame["feature"].nunique()]
            if col is None
            else [
                frame.query(f"{col} == '{col_value}'")["feature"].nunique()
                for col_value in frame[col].unique()
            ]
        )
        for ax, n_feature in zip(ax_grid.flatten(), n_features, strict=True):
            _decorate_matplotlib_axis(
                ax=ax,
                add_background_features=add_background_features,
                n_features=n_feature,
                xlabel="Mean decrease in impurity",
                ylabel="",
            )
        return figure

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
    ) -> Figure:
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

        figure = self._categorical_plot(
            frame=frame,
            report_type=report_type,
            hue=hue,
            col=col,
            barplot_kwargs={"sharey": has_same_features} | barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs={"sharey": has_same_features} | stripplot_kwargs,
        )
        figure.suptitle("Mean decrease in impurity (MDI)")
        return figure

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
        None

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
