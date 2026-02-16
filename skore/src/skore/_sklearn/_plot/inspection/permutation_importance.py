from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike
from scipy.sparse import issparse, spmatrix
from sklearn.base import BaseEstimator, is_classifier
from sklearn.inspection import permutation_importance
from sklearn.metrics._scorer import _BaseScorer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import _num_features

from skore._sklearn._plot.base import BOXPLOT_STYLE, DisplayMixin
from skore._sklearn._plot.inspection.utils import _decorate_matplotlib_axis
from skore._sklearn.feature_names import _get_feature_names
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
        - `split`
        - `feature`
        - `label` or `output` (classification vs. regression)
        - `repetition`
        - `value`

    report_type : {"estimator", "cross-validation"}
        Report type from which the display is created.

    Attributes
    ----------
    facet_ : seaborn FacetGrid
        FacetGrid containing the permutation importance.

    figure_ : matplotlib Figure
        Figure containing the permutation importance.

    ax_ : matplotlib Axes
        Axes with permutation importance.
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
        cls,
        *,
        data_source: DataSource,
        estimators: Sequence[BaseEstimator],
        estimator_names: Sequence[str],
        splits: Sequence[int | float],
        Xs: Sequence[ArrayLike],
        ys: Sequence[ArrayLike],
        at_step: int | str,
        metric: str | Callable | list[str] | tuple[str] | dict[str, Callable] | None,
        n_repeats: int,
        max_samples: float,
        n_jobs: int | None,
        seed: int | None,
        report_type: ReportType,
    ) -> "PermutationImportanceDisplay":
        if not isinstance(at_step, str | int):
            raise ValueError(f"at_step must be an integer or a string; got {at_step!r}")

        if isinstance(at_step, str):
            first_estimator = estimators[0]
            if not isinstance(first_estimator, Pipeline):
                raise ValueError(
                    "at_step can only be a string when the estimator is a Pipeline"
                )
            at_step = list(first_estimator.named_steps.keys()).index(at_step)

        all_importances = []
        for estimator, estimator_name, split, X, y in zip(
            estimators, estimator_names, splits, Xs, ys, strict=True
        ):
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
            df_importances["estimator"] = estimator_name
            df_importances["split"] = split
            all_importances.append(df_importances)

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
            importances=pd.concat(all_importances, axis="index")[ordered_columns],
            report_type=report_type,
        )

    @staticmethod
    def _get_columns_to_groupby(*, frame: pd.DataFrame) -> list[str]:
        """Get the available columns from which to group by."""
        columns_to_groupby = list[str]()
        if "label" in frame.columns and frame["label"].nunique() > 1:
            columns_to_groupby.append("label")
        if "output" in frame.columns and frame["output"].nunique() > 1:
            columns_to_groupby.append("output")
        if "split" in frame.columns and frame["split"].nunique() > 1:
            columns_to_groupby.append("split")
        return columns_to_groupby

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        metric: str | None = None,
        subplot_by: str | tuple[str, ...] | None = "auto",
    ) -> None:
        """Plot the permutation importance.

        Parameters
        ----------
        metric : str or None, default=None
            Metric to plot.

        subplot_by : str, tuple of str or None, default="auto"
            Column(s) to use for subplotting. The possible values are:

            - if `"auto"`, depending of the information available, a meaningful decision
              is made to create subplots.
            - if a string, the corresponding column of the dataframe is used to create
              several subplots. Those plots will be a organized in a grid of a single
              row and several columns.
            - if a tuple of 2 strings, the corresponding columns are used to create
              subplots in a grid. The first element is the row, the second is the
              column.
            - if `None`, all information is plotted on a single plot. An error is raised
              if there is too much information to plot on a single plot.

        """
        return self._plot(subplot_by=subplot_by, metric=metric)

    def _plot_matplotlib(
        self,
        *,
        metric: str,
        subplot_by: str | tuple[str, ...] | None = "auto",
    ) -> None:
        """Dispatch the plotting function for matplotlib backend."""
        boxplot_kwargs = self._default_boxplot_kwargs.copy()
        stripplot_kwargs = self._default_stripplot_kwargs.copy()
        frame = self.frame(metric=metric, aggregate=None)

        self._plot_single_estimator(
            subplot_by=subplot_by,
            frame=frame,
            estimator_name=self.importances["estimator"].unique()[0],
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
        )

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

    def _plot_single_estimator(
        self,
        *,
        subplot_by: str | tuple[str, ...] | None,
        frame: pd.DataFrame,
        estimator_name: str,
        boxplot_kwargs: dict[str, Any],
        stripplot_kwargs: dict[str, Any],
    ) -> None:
        """Plot the permutation importance for an `EstimatorReport`."""
        aggregate_title = ""
        if subplot_by == "auto" or subplot_by is None:
            columns_to_groupby = self._get_columns_to_groupby(frame=frame)

            if "split" in columns_to_groupby:
                frame = self._aggregate_over_split(frame=frame)
                columns_to_groupby.remove("split")
                aggregate_title = "averaged over splits"

            if subplot_by == "auto":
                col = columns_to_groupby[0] if columns_to_groupby else None
                hue, row = None, None
            else:  # subplot_by is None
                hue = columns_to_groupby[0] if columns_to_groupby else None
                col, row = None, None

        else:
            columns_to_groupby = self._get_columns_to_groupby(frame=frame)
            subplot_cols = (subplot_by,) if isinstance(subplot_by, str) else subplot_by
            invalid = [c for c in subplot_cols if c not in columns_to_groupby]
            if invalid:
                raise ValueError(
                    f"The column(s) {invalid} are not available. You can use the "
                    "following values to create subplots: "
                    f"{', '.join(columns_to_groupby)}"
                )

            remaining = set(columns_to_groupby) - set(subplot_cols)
            if "split" in remaining:
                frame = self._aggregate_over_split(frame=frame)
                remaining.remove("split")
                aggregate_title = "averaged over splits"

            match len(subplot_cols):
                case 1:
                    row, col, hue = (
                        None,
                        subplot_cols[0],
                        next(iter(remaining), None),
                    )
                case 2:
                    row, col, hue = (
                        subplot_cols[0],
                        subplot_cols[1],
                        next(iter(remaining), None),
                    )
                case _:
                    raise ValueError(
                        "Expected 1 to 2 columns for subplot_by, got "
                        f"{len(subplot_cols)}. You can use the following values: "
                        f"{', '.join(columns_to_groupby)}"
                    )

        if hue is None:
            # we don't need the palette and we are at risk of raising an error or
            # deprecation warning if passing palette without a hue
            stripplot_kwargs.pop("palette", None)
        else:
            boxplot_kwargs.setdefault("palette", "tab10")

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

        metric_name = frame["metric"].unique()[0]
        self.figure_, self.ax_ = self.facet_.figure, self.facet_.axes.squeeze()
        for row_axes in self.facet_.axes:
            for ax in row_axes:
                _decorate_matplotlib_axis(
                    ax=ax,
                    add_background_features=add_background_features,
                    n_features=frame["feature"].nunique(),
                    xlabel=f"Decrease in {metric_name}",
                    ylabel="",
                )
        if len(self.ax_.flatten()) == 1:
            self.ax_ = self.ax_.flatten()[0]
        data_source = frame["data_source"].unique()[0]
        self.figure_.suptitle(
            f"Permutation importance {aggregate_title} \n"
            f"of {estimator_name} on {data_source} set"
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
            columns_to_drop = ["estimator", "split"]
            group_by = ["data_source", "metric", "feature"]
        elif self.report_type == "cross-validation":
            columns_to_drop = ["estimator"]
            group_by = ["data_source", "metric", "split", "feature"]
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
                # output or labels in case of mixed metrics (i.e. averaged vs.
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
            boxplot_kwargs=boxplot_kwargs or {},
            stripplot_kwargs=stripplot_kwargs or {},
        )
