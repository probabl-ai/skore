from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, is_classifier
from sklearn.inspection import permutation_importance
from sklearn.metrics._scorer import _BaseScorer

from skore._sklearn._plot.base import BOXPLOT_STYLE, DisplayMixin
from skore._sklearn._plot.feature_importance.utils import _decorate_matplotlib_axis
from skore._sklearn.types import Aggregate, DataSource, ReportType
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
        - `feature`
        - `label` or `output` (classification vs. regression)
        - `repetition`
        - `value`

    report_type : {"estimator"}
        Report type from which the display is created.
    """

    _default_boxplot_kwargs: dict[str, Any] = {
        "whis": 1e10,
        **BOXPLOT_STYLE,
    }
    _default_stripplot_kwargs: dict[str, Any] = {"palette": "tab10", "alpha": 0.5}

    def __init__(self, *, importances: pd.DataFrame, report_type: ReportType):
        self.importances = importances
        self.report_type = report_type

    @classmethod
    def _compute_data_for_display(
        self,
        *,
        data_source: DataSource,
        estimator: BaseEstimator,
        estimator_name: str,
        X: ArrayLike,
        y: ArrayLike,
        feature_names: list[str],
        metric: str | Callable | list[str] | tuple[str] | dict[str, Callable] | None,
        n_repeats: int,
        max_samples: float,
        n_jobs: int | None,
        seed: int | None,
        report_type: ReportType,
    ) -> "PermutationImportanceDisplay":
        scores = permutation_importance(
            estimator=estimator,
            X=X,
            y=y,
            scoring=metric,
            n_repeats=n_repeats,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=seed,
        )

        if "importances" in scores:
            # single metric case -> switch to multi-metric case by wrapping in a dict
            # with the name of the metric
            metric = cast(str | Callable | _BaseScorer | None, metric)
            if metric is None:
                metric_name = "accuracy" if is_classifier(estimator) else "r2"
            elif isinstance(metric, str):
                metric_name = metric
            elif isinstance(metric, _BaseScorer):
                metric_name = metric._score_func.__name__.replace("_", " ")
            else:
                metric_name = metric.__name__.replace("_", " ")
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

                if metric_importances.shape[-1] == 1:  # scalar metric
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

        ordered_columns = [
            "estimator",
            "data_source",
            "metric",
            "feature",
            "label",
            "output",
            "repetition",
            "value",
        ]
        df_importances = pd.concat(df_importances, axis="index")
        df_importances["data_source"] = data_source
        df_importances["estimator"] = estimator_name

        return PermutationImportanceDisplay(
            importances=df_importances[ordered_columns], report_type=report_type
        )

    @staticmethod
    def _get_columns_to_groupby(*, frame: pd.DataFrame) -> list[str]:
        """Get the available columns from which to group by."""
        columns_to_groupby = list[str]()
        if "metric" in frame.columns and frame["metric"].nunique() > 1:
            columns_to_groupby.append("metric")
        if "label" in frame.columns and frame["label"].nunique() > 1:
            columns_to_groupby.append("label")
        if "output" in frame.columns and frame["output"].nunique() > 1:
            columns_to_groupby.append("output")
        return columns_to_groupby

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        subplot_by: str | tuple[str, str] | None = "auto",
        metric: str | list[str] | None = None,
    ) -> None:
        """Plot the permutation importance.

        Parameters
        ----------
        subplot_by : str, tuple of str or None, default="auto"
            Column(s) to use for subplotting. The possible values are:

            - if `"auto"`, depending of the information available, a meaningful decision
              is made to create subplots.
            - if a string, the corresponding column of the dataframe is used to create
              several subplots. Those plots will be a organized in a grid of a single
              row and several columns.
            - if a tuple of strings, the corresponding columns of the dataframe are used
              to create several subplots. Those plots will be a organized in a grid of
              several rows and columns. The first element of the tuple is the row and
              the second element is the column.
            - if `None`, all information is plotted on a single plot. An error is raised
              if there is too much information to plot on a single plot.

        metric : str or list of str, default=None
            Filter the importances by metric. If `None`, all importances associated with
            each metric are plotted.
        """
        return self._plot(subplot_by=subplot_by, metric=metric)

    def _plot_matplotlib(
        self,
        *,
        subplot_by: str | tuple[str, str] | None = "auto",
        metric: str | list[str] | None = None,
    ) -> None:
        """Dispatch the plotting function for matplotlib backend."""
        boxplot_kwargs = self._default_boxplot_kwargs.copy()
        stripplot_kwargs = self._default_stripplot_kwargs.copy()
        frame = self.frame(metric=metric, aggregate=None)

        err_msg = (
            "You try to plot the permutation importance of metrics averaged over {} "
            "and other without averaging. This setting is not supported. Please filter "
            "a group of consistent metrics using the `metric` parameter."
        )
        if "label" in frame.columns and frame["label"].isna().any():
            raise ValueError(err_msg.format("labels"))
        elif "output" in frame.columns and frame["output"].isna().any():
            raise ValueError(err_msg.format("outputs"))

        self._plot_single_estimator(
            subplot_by=subplot_by,
            frame=frame,
            estimator_name=self.importances["estimator"].unique()[0],
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
        )

    def _plot_single_estimator(
        self,
        *,
        subplot_by: str | tuple[str, str] | None,
        frame: pd.DataFrame,
        estimator_name: str,
        boxplot_kwargs: dict[str, Any],
        stripplot_kwargs: dict[str, Any],
    ) -> None:
        """Plot the permutation importance for an `EstimatorReport`."""
        if subplot_by == "auto":
            is_multi_metric = frame["metric"].nunique() > 1
            is_multi_target = any(name in frame.columns for name in ["label", "output"])
            if is_multi_metric and is_multi_target:
                hue, col, row = (
                    "label" if "label" in frame.columns else "output",
                    "metric",
                    None,
                )
            elif is_multi_metric:
                hue, col, row = None, "metric", None
            elif is_multi_target:
                hue, col, row = (
                    "label" if "label" in frame.columns else "output",
                    None,
                    None,
                )
            else:
                hue, col, row = None, None, None
        elif subplot_by is None:
            # Possible accepted values: {"metric"}, {"label"}, {"output"}
            columns_to_groupby = self._get_columns_to_groupby(frame=frame)
            n_columns_to_groupby = len(columns_to_groupby)
            if n_columns_to_groupby > 1:
                raise ValueError(
                    "Cannot plot all the available information available on a single "
                    "plot. Please set `subplot_by` to a string or a tuple of strings. "
                    "You can use the following values to create subplots: "
                    f"{', '.join(columns_to_groupby)}"
                )
            elif n_columns_to_groupby == 1:
                hue, col, row = columns_to_groupby[0], None, None
            else:
                hue, col, row = None, None, None
        else:
            # Possible accepted values: {"metric"}, {"metric", "label"},
            # {"metric", "output"}
            columns_to_groupby = self._get_columns_to_groupby(frame=frame)
            if isinstance(subplot_by, str):
                if subplot_by not in columns_to_groupby:
                    raise ValueError(
                        f"The column {subplot_by} is not available. You can use the "
                        "following values to create subplots: "
                        f"{', '.join(columns_to_groupby)}"
                    )
                col, row = subplot_by, None
                if remaining_column := set(columns_to_groupby) - {subplot_by}:
                    hue = next(iter(remaining_column))
                else:
                    hue = None
            else:
                if not all(item in columns_to_groupby for item in subplot_by):
                    raise ValueError(
                        f"The columns {subplot_by} are not available. You can use the "
                        "following values to create subplots: "
                        f"{', '.join(columns_to_groupby)}"
                    )
                (row, col), hue = subplot_by, None

        if hue is None:
            # we don't need the palette and we are at risk of raising an error or
            # deprecation warning if passing palette without a hue
            stripplot_kwargs.pop("palette", None)

        self.facet_ = sns.catplot(
            data=frame,
            x="value",
            y="feature",
            hue=hue,
            col=col,
            row=row,
            kind="strip",
            dodge=True,
            sharex=False,
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

        metrics = frame["metric"].unique()
        self.figure_, self.ax_ = self.facet_.figure, self.facet_.axes.squeeze()
        for row_index, row_axes in enumerate(self.facet_.axes):
            for col_index, ax in enumerate(row_axes):
                if len(metrics) > 1:
                    if row == "metric":
                        xlabel = f"Decrease in {metrics[row_index]}"
                    elif col == "metric":
                        xlabel = f"Decrease in {metrics[col_index]}"
                    else:
                        xlabel = "Decrease in metric"
                else:
                    xlabel = f"Decrease in {metrics[0]}"

                _decorate_matplotlib_axis(
                    ax=ax,
                    add_background_features=add_background_features,
                    n_features=frame["feature"].nunique(),
                    xlabel=xlabel,
                    ylabel="",
                )
        if len(self.ax_.flatten()) == 1:
            self.ax_ = self.ax_.flatten()[0]
        data_source = frame["data_source"].unique()[0]
        self.figure_.suptitle(
            f"Permutation importance of {estimator_name} on {data_source} set"
        )

    def frame(
        self,
        *,
        metric: str | list[str] | None = None,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Get the feature importance in a dataframe format.

        Parameters
        ----------
        metric : str or list of str, default=None
            Filter the importances by metric. If `None`, all importances associated with
            each metric are returned.

        aggregate : {"mean", "std"}, ("mean", std) or None, default=("mean", "std")
            Aggregate the importances by the given metric.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the importances.
        """
        if self.report_type == "estimator":
            columns_to_drop = ["estimator"]
            group_by = ["data_source", "metric", "feature"]
        else:
            raise TypeError(f"Unexpected report type: {self.report_type!r}")

        frame = self.importances.copy()
        if metric is not None:
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

        if aggregate is not None:
            frame = (
                frame.drop(columns=["repetition"])
                # avoid sorting the features by name and do not drop NA from
                # output or labels in case of mixed metrics (i.e. averaged vs\
                # non-averaged)
                .groupby(group_by, sort=False, dropna=False)
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
        self : object
            Returns the instance itself.

        Raises
        ------
        ValueError
            If a style parameter is unknown.
        """
        return super().set_style(
            policy=policy,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
        )
