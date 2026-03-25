from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy.sparse import issparse, spmatrix
from sklearn.base import BaseEstimator, is_classifier
from sklearn.inspection import permutation_importance
from sklearn.metrics._scorer import _BaseScorer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import _num_features

from skore._sklearn._plot.base import BOXPLOT_STYLE, DisplayMixin
from skore._sklearn._plot.inspection.utils import (
    _decorate_matplotlib_axis,
    select_k_features,
    sort_features,
)
from skore._sklearn.feature_names import _get_feature_names
from skore._sklearn.types import Aggregate, DataSource, Metric, ReportType
from skore._utils._index import flatten_multi_index


class PermutationImportanceDisplay(DisplayMixin):
    """Display to inspect feature importance via feature permutation.

    Parameters
    ----------
    importances : pd.DataFrame
        The importances computed after permuting the input features. The columns are:

        - `estimator`
        - `data_source`
        - `metric`
        - `split`
        - `feature`
        - `label` or `output` (classification vs. regression)
        - `repetition`
        - `value`

    report_type : {"estimator", "cross-validation", "comparison-estimator", \
            "comparison-cross-validation"}
        Report type from which the display is created.
    """

    _default_boxplot_kwargs: dict[str, Any] = {
        "whis": 1e10,
        **BOXPLOT_STYLE,
    }
    _default_stripplot_kwargs: dict[str, Any] = {
        "palette": "tab10",
        "alpha": 0.5,
        "height": 6,
    }

    def __init__(self, *, importances: pd.DataFrame, report_type: ReportType):
        self.importances = importances
        self.report_type = report_type

    @classmethod
    def _compute_data_for_display(
        cls,
        *,
        data_source: DataSource,
        estimator: BaseEstimator,
        name: str,
        X: ArrayLike,
        y: ArrayLike,
        at_step: int | str,
        metric: Metric | list[Metric] | dict[str, Metric] | None,
        n_repeats: int,
        max_samples: float,
        n_jobs: int | None,
        seed: int | None,
        report_type: ReportType,
    ) -> PermutationImportanceDisplay:
        if not isinstance(at_step, str | int):
            raise ValueError(f"at_step must be an integer or a string; got {at_step!r}")

        if isinstance(at_step, str):
            if not isinstance(estimator, Pipeline):
                raise ValueError(
                    "at_step can only be a string when the estimator is a Pipeline"
                )
            at_step = list(estimator.named_steps.keys()).index(at_step)

        if isinstance(estimator, Pipeline) and at_step != 0:
            if abs(at_step) >= len(estimator.steps):
                raise ValueError(
                    "at_step must be strictly smaller in magnitude than the "
                    "number of steps in the Pipeline, which is "
                    f"{len(estimator.steps)}; got {at_step}"
                )
            feature_engineering = estimator[:at_step]
            estimator = estimator[at_step:]
            X_transformed = feature_engineering.transform(X)
        else:
            feature_engineering = None
            X_transformed = X

        feature_names = _get_feature_names(
            estimator,
            n_features=_num_features(X_transformed),
            X=X_transformed,
            transformer=feature_engineering,
        )

        if issparse(X_transformed):
            X_transformed = cast(spmatrix, X_transformed)
            X_transformed = X_transformed.todense()

        scores = permutation_importance(
            estimator=estimator,
            X=X_transformed,
            y=y,
            scoring=metric,
            n_repeats=n_repeats,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=seed,
        )

        if "importances" in scores:
            # single metric case -> switch to multi-metric case by wrapping in a
            # dict with the name of the metric
            metric_obj = cast(str | Callable | _BaseScorer | None, metric)
            if metric_obj is None:
                metric_name = "accuracy" if is_classifier(estimator) else "r2"
            elif isinstance(metric_obj, str):
                metric_name = metric_obj
            elif isinstance(metric_obj, _BaseScorer):
                metric_name = metric_obj._score_func.__name__.replace("_", " ")
            else:
                metric_name = metric_obj.__name__.replace("_", " ")
            scores = {metric_name: scores}

        df_importances = []
        for metric_name, metric_values in scores.items():
            metric_importances = np.atleast_3d(metric_values["importances"])

            df_metric_importances = []
            # we loop across the labels (for classification) or the outputs
            # (for regression)
            for target_index, target_importances in enumerate(
                np.moveaxis(metric_importances, -1, 0)
            ):
                df = pd.DataFrame(
                    target_importances,
                    index=feature_names,
                    columns=range(1, n_repeats + 1),
                ).melt(var_name="repetition")

                if metric_importances.shape[-1] == 1:
                    df["label"], df["output"] = np.nan, np.nan
                else:
                    if is_classifier(estimator):
                        df["label"] = estimator.classes_[target_index]
                        df["output"] = np.nan
                    else:
                        df["output"], df["label"] = target_index, np.nan

                df["metric"] = metric_name
                df["feature"] = np.tile(feature_names, n_repeats)
                df_metric_importances.append(df)

            df_metric_importances = pd.concat(df_metric_importances, axis="index")
            df_importances.append(df_metric_importances)

        df_importances = pd.concat(df_importances, axis="index")
        df_importances["data_source"] = data_source
        df_importances["estimator"] = name
        df_importances["split"] = np.nan

        ordered_columns = [
            "estimator",
            "data_source",
            "metric",
            "split",
            "feature",
            "label",
            "output",
            "repetition",
            "value",
        ]

        return cls(
            importances=df_importances.reset_index(drop=True)[ordered_columns],
            report_type=report_type,
        )

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        metric: str | None = None,
        subplot_by: str | None = "auto",
        select_k: int | None = None,
        sorting_order: Literal["descending", "ascending", None] = None,
    ) -> Figure:
        """Plot the permutation importance.

        Parameters
        ----------
        metric : str or None, default=None
            Metric to plot. Required when multiple metrics are computed.

        subplot_by : str or None, default="auto"
            Column to use for subplotting. The possible values are:

            - if `"auto"`, depending of the information available, a meaningful decision
              is made to create subplots.
            - if a string, the corresponding column of the dataframe is used to create
              several subplots. Those plots will be organized in a grid of a single
              row and several columns.
            - if `None`, all information is plotted on a single plot. An error is raised
              if there is too much information to plot on a single plot.

        select_k : int, default=None
            If set, only the top (positive) or bottom (negative) k features by
            importance are shown. See :meth:`frame` for details.

        sorting_order : {"descending", "ascending", None}, default=None
            Sort features by importance before plotting. See :meth:`frame` for details.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the permutation importance plot.
        """
        return self._plot(
            subplot_by=subplot_by,
            metric=metric,
            select_k=select_k,
            sorting_order=sorting_order,
        )

    def _plot_matplotlib(
        self,
        *,
        metric: str | None = None,
        subplot_by: str | None = "auto",
        select_k: int | None = None,
        sorting_order: Literal["descending", "ascending", None] = None,
    ) -> Figure:
        """Dispatch the plotting function for matplotlib backend."""
        if select_k == 0:
            raise ValueError(
                "select_k=0 would produce an empty plot. Use a non-zero value or "
                "omit select_k to plot all features."
            )
        boxplot_kwargs = self._default_boxplot_kwargs.copy()
        stripplot_kwargs = self._default_stripplot_kwargs.copy()

        frame = self.frame(
            metric=metric,
            aggregate=None,
            select_k=select_k,
            sorting_order=sorting_order,
        )
        columns_to_groupby = self._get_columns_to_groupby(frame=frame)
        has_same_features = (
            self._has_same_features(frame=frame)
            if "comparison" in self.report_type
            else True
        )

        if metric is None and self.importances["metric"].nunique() > 1:
            raise ValueError(
                "Multiple metrics cannot be plotted at once. Please select a metric"
                " to plot the associated importances using the `metric` parameter."
            )

        if subplot_by not in ("auto", None) and subplot_by not in columns_to_groupby:
            raise ValueError(
                f"The column {subplot_by!r} is not available for subplotting. "
                "You can use the following values to create subplots: "
                f"{', '.join(columns_to_groupby + ['auto', 'None'])}"
            )

        if subplot_by == "auto":
            subplot_by = (
                "estimator"
                if "comparison" in self.report_type
                and (
                    "label" in columns_to_groupby
                    or "output" in columns_to_groupby
                    or not has_same_features
                )
                else None
            )

        remaining = [col for col in columns_to_groupby if col != subplot_by]
        if "split" in remaining:
            frame = self._aggregate_over_split(frame=frame)
            remaining.remove("split")
            aggregate_info = "averaged over splits,"
        else:
            aggregate_info = ""
        hue = remaining[0] if remaining else None
        col = subplot_by

        if not has_same_features and hue == "estimator":
            raise ValueError(
                "The estimators have different features and should be plotted on "
                "different axis using `subplot_by='estimator'`."
            )

        if hue is None:
            stripplot_kwargs.pop("palette", None)
            # we don't need the palette and we are at risk of raising an error or
            # deprecation warning if passing palette without a hue

        figure = self._categorical_plot(
            frame=frame,
            hue=hue,
            col=col,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
            sharey=has_same_features,
        )

        data_source = frame["data_source"].unique()[0]
        estimator_info = (
            ""
            if "comparison" in self.report_type
            else f" of {self.importances['estimator'].unique()[0]}"
        )
        figure.suptitle(
            f"Permutation importance{estimator_info}\n"
            f"{aggregate_info} on {data_source} set"
        )
        return figure

    @staticmethod
    def _get_columns_to_groupby(*, frame: pd.DataFrame) -> list[str]:
        """Get the available columns from which to group by."""
        columns_to_groupby = list[str]()
        if "estimator" in frame.columns and frame["estimator"].nunique() > 1:
            columns_to_groupby.append("estimator")
        if "label" in frame.columns and frame["label"].nunique() > 1:
            columns_to_groupby.append("label")
        if "output" in frame.columns and frame["output"].nunique() > 1:
            columns_to_groupby.append("output")
        if "split" in frame.columns and frame["split"].nunique() > 1:
            columns_to_groupby.append("split")
        return columns_to_groupby

    @staticmethod
    def _aggregate_over_split(*, frame: pd.DataFrame) -> pd.DataFrame:
        """Compute the averaged scores over the splits."""
        group_by = frame.columns.difference(["repetition", "value"]).tolist()
        return (
            frame.drop(columns=["repetition"])
            .groupby(group_by, sort=False, dropna=False)
            .aggregate("mean")
            .reset_index()
        )

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

    def _categorical_plot(
        self,
        *,
        frame: pd.DataFrame,
        hue: str | None,
        col: str | None,
        boxplot_kwargs: dict[str, Any],
        stripplot_kwargs: dict[str, Any],
        sharey: bool,
    ) -> Figure:
        """Plot importances with strip + box overlays.

        Parameters
        ----------
        frame : pd.DataFrame
            Dataframe containing permutation importances to display.

        hue : str or None
            Column used to group values with colors.

        col : str or None
            Column used to create subplot columns.

        boxplot_kwargs : dict
            Keyword arguments forwarded to :func:`seaborn.boxplot`.

        stripplot_kwargs : dict
            Keyword arguments forwarded to :func:`seaborn.stripplot`.

        sharey : bool
            Whether feature axes are shared across subplot columns.
        """
        # Ensure seaborn receives a clean, unique index when concatenating groups.
        frame = frame.reset_index(drop=True)
        facet = sns.catplot(
            data=frame,
            x="value",
            y="feature",
            hue=hue,
            col=col,
            kind="strip",
            dodge=True,
            sharex=False,
            sharey=sharey,
            **stripplot_kwargs,
        ).map_dataframe(
            sns.boxplot,
            x="value",
            y="feature",
            hue=hue,
            dodge=True,
            **boxplot_kwargs,
        )
        add_background_features = hue is not None
        metric_name = frame["metric"].unique()[0]
        figure = facet.figure
        ax_grid = facet.axes.squeeze()
        n_features = (
            [frame["feature"].nunique()]
            if col is None
            else [
                frame[frame[col] == col_value]["feature"].nunique()
                for col_value in frame[col].unique()
            ]
        )
        for ax, n_feature in zip(ax_grid.flatten(), n_features, strict=True):
            _decorate_matplotlib_axis(
                ax=ax,
                add_background_features=add_background_features,
                n_features=n_feature,
                xlabel=f"Decrease in {metric_name}",
                ylabel="",
            )
        return figure

    def frame(
        self,
        *,
        metric: str | list[str] | None = None,
        aggregate: Aggregate | None = ("mean", "std"),
        level: Literal["splits", "repetitions"] = "splits",
        select_k: int | None = None,
        sorting_order: Literal["descending", "ascending", None] = None,
    ) -> pd.DataFrame:
        """Get the feature importance in a dataframe format.

        Parameters
        ----------
        metric : str or list of str, default=None
            Filter the importances by metric. If `None`, all importances associated with
            each metric are returned.

        aggregate : {"mean", "std"}, ("mean", std) or None, default=("mean", "std")
            How to aggregate the importances. Applied on repetitions or on repetitions
            then splits, for (comparisons of) cross validation reports and depending
            on the value of `level`.

        level : {"splits", "repetitions"}, default="splits"
            Over which dimensions to aggregate when `aggregate` is not `None`.
            `"repetitions"` aggregates only over repetitions (keeps `split` for
            cross-validation). `"splits"` aggregates over repetitions then over
            splits. Only relevant when `aggregate` is not `None` and the report is
            a :class:`~skore.CrossValidationReport` or a
            :class:`~skore.ComparisonReport` containing such type of report.

        select_k : int, default=None
            Select features by importance:

            - Positive values: the `select_k` features with largest importance
            - Negative values: the `-select_k` features with smallest importance

            Selection is performed independently within each group (estimator, and
            per label/output if applicable). For cross-validation, features are
            ranked by mean importance across splits. When ``aggregate`` is
            ``None``, ranking uses mean importance per feature over repetitions
            (and splits); all repetition/split rows are kept for the selected
            features.

        sorting_order : {"descending", "ascending", None}, default=None
            Sort features by importance: "descending" (most important first),
            "ascending" (least important first), or None to preserve original order.
            Can be used independently of `select_k`. When ``aggregate`` is ``None``,
            ordering uses mean importance per feature over repetitions (and splits).

        Returns
        -------
        pd.DataFrame
            The feature importances. The columns depend on the
            report type and parameters, and include:

            - `data_source`: Data source used to compute the importances
              (``"train"`` or ``"test"``).
            - `metric`: Metric used to compute the importances.
            - `feature`: Feature name.
            - `value_mean` and `value_std`: Aggregated importance values
              (only when ``aggregate`` is not ``None``).
            - `value`: Raw importance value per repetition
              (only when ``aggregate`` is ``None``).
            - `estimator`: Name of the estimator (for comparison reports).
            - `split`: Cross-validation split index (for cross-validation
              reports, only when ``aggregate`` is ``None``).
            - `label`: Class label (for multiclass classification).
            - `output`: Output index (for multi-output regression).
        """
        if self.report_type == "estimator":
            columns_to_drop = ["estimator", "split"]
            group_by = ["data_source", "metric", "feature"]
        elif self.report_type == "cross-validation":
            columns_to_drop = ["estimator"]
            group_by = ["data_source", "metric", "split", "feature"]
        elif self.report_type == "comparison-estimator":
            columns_to_drop = ["split"]
            group_by = ["estimator", "data_source", "metric", "feature"]
        else:  # comparison-cross-validation
            columns_to_drop = []
            group_by = ["estimator", "data_source", "metric", "split", "feature"]

        frame = self.importances.copy()
        if metric is not None:
            if isinstance(metric, str):
                metric = [metric]
            available_metrics = self.importances["metric"].unique()
            for m in metric:
                if m not in available_metrics:
                    raise ValueError(
                        f"The metric {m!r} is not available. Please select metrics "
                        f"from the following list: {', '.join(available_metrics)}. "
                        "Otherwise, use the `metric` parameter of the "
                        "`.permutation_importance()` method to specify the metrics to "
                        "use for computing the importances."
                    )
            frame = frame.query("metric in @metric")
        if frame["label"].isna().all():
            # regression problem or averaged classification metric
            columns_to_drop.append("label")
        else:
            group_by.append("label")
        if frame["output"].isna().all():
            # classification problem or averaged regression metric
            columns_to_drop.append("output")
        else:
            group_by.append("output")

        frame = frame.drop(columns=columns_to_drop)

        if sorting_order is not None:
            frame = sort_features(
                frame,
                sorting_order,
                group_columns=[
                    c for c in self._get_columns_to_groupby(frame=frame) if c != "split"
                ],
                importance_column="value",
            )
        if select_k is not None:
            frame = select_k_features(
                frame,
                select_k,
                group_columns=[
                    c for c in self._get_columns_to_groupby(frame=frame) if c != "split"
                ],
                importance_column="value",
            )

        if aggregate is not None:
            if level not in ("splits", "repetitions"):
                raise ValueError(
                    f"Invalid value for `level`: {level!r}. Valid values are "
                    "`'splits'` and `'repetitions'`."
                )
            frame = (
                frame.drop(columns=["repetition"])
                # avoid sorting the features by name and do not drop NA from
                # output or labels in case of mixed metrics (i.e. averaged vs.
                # non-averaged)
                .groupby(group_by, sort=False, dropna=False)
                .aggregate(aggregate)
            ).reset_index()
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = flatten_multi_index(frame.columns)

            if level == "splits" and "split" in frame.columns:
                columns_to_drop = ["split"]
                if "value_std" in frame.columns:
                    columns_to_drop.append("value_std")
                frame = (
                    frame.drop(columns=columns_to_drop)
                    .rename(columns={"value_mean": "value"})
                    .groupby(
                        [c for c in group_by if c not in columns_to_drop],
                        sort=False,
                        dropna=False,
                    )
                    .aggregate(aggregate)
                ).reset_index()
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = flatten_multi_index(frame.columns)

        return frame

    # ignore the type signature because we override kwargs by specifying the name of
    # the parameters for the user.
    def set_style(  # type: ignore[override]
        self,
        *,
        policy: Literal["override", "update"] = "update",
        boxplot_kwargs: dict[str, Any] | None = None,
        stripplot_kwargs: dict[str, Any] | None = None,
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : {"override", "update"}, default="update"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        boxplot_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`seaborn.boxplot` for
            rendering the importances with a :class:`~skore.EstimatorReport`.

        stripplot_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`seaborn.stripplot` for
            rendering the importances with a :class:`~skore.EstimatorReport`.

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
            boxplot_kwargs=boxplot_kwargs or {},
            stripplot_kwargs=stripplot_kwargs or {},
        )
