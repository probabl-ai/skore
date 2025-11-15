from collections.abc import Callable, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend import Legend
from sklearn.base import BaseEstimator, is_classifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from skore._sklearn._plot.base import BOXPLOT_STYLE, DisplayMixin
from skore._sklearn._plot.utils import _despine_matplotlib_axis
from skore._sklearn.feature_names import _get_feature_names
from skore._sklearn.types import ReportType


class CoefficientsDisplay(DisplayMixin):
    """Display to inspect the coefficients of linear models.

    Parameters
    ----------
    coefficients : DataFrame | list[DataFrame]
        The coefficients data to display.

    report_type : {"estimator", "cross-validation", "comparison-estimator", \
            "comparison-cross-validation"}
        Report type from which the display is created.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with the different matplotlib axis.

    figure_ : matplotlib Figure
        Figure containing the plot.
    """

    def __init__(self, *, coefficients: pd.DataFrame, report_type: ReportType):
        self.coefficients = coefficients
        self.report_type = report_type

    def frame(self):
        """Get the coefficients in a dataframe format.

        Returns
        -------
        DataFrame
            Dataframe containing the coefficients of the linear model.
        """
        if self.report_type == "estimator":
            columns_to_drop = ["estimator", "split"]
        elif self.report_type == "cross-validation":
            columns_to_drop = ["estimator"]
        elif self.report_type == "comparison-estimator":
            columns_to_drop = ["split"]
        elif self.report_type == "comparison-cross-validation":
            columns_to_drop = []
        else:
            raise TypeError(f"Unexpected report type: {self.report_type!r}")

        if self.coefficients["label"].isna().all():
            # regression problem
            columns_to_drop.append("label")
        if self.coefficients["output"].isna().all():
            # classification problem
            columns_to_drop.append("output")

        return self.coefficients.drop(columns=columns_to_drop)

    @DisplayMixin.style_plot
    def plot(
        self, *, subplots_by: Literal["estimator", "label", "output"] | None = None
    ) -> None:
        return self._plot(subplots_by=subplots_by)

    def _plot_matplotlib(
        self, *, subplots_by: Literal["estimator", "label", "output"] | None = None
    ):
        if self.report_type == "estimator":
            return self._plot_estimator_report(subplots_by=subplots_by)
        elif self.report_type == "cross-validation":
            return self._plot_cross_validation_report(subplots_by=subplots_by)
        elif self.report_type == "comparison-estimator":
            return self._plot_comparison_report_estimator(subplots_by=subplots_by)
        elif self.report_type == "comparison-cross-validation":
            return self._plot_comparison_report_cross_validation(
                subplots_by=subplots_by
            )
        else:
            raise TypeError(f"Unexpected report type: {self.report_type!r}")

    @staticmethod
    def _decorate_matplotlib_axis(*, ax: plt.Axes):
        ax.axvline(x=0, color=".5", linestyle="--")
        ax.set(xlabel="Magnitude of coefficient", ylabel="")
        _despine_matplotlib_axis(
            ax,
            axis_to_despine=("top", "right", "left"),
            remove_ticks=True,
            x_range=None,
            y_range=None,
        )

    @staticmethod
    def _set_legend(*, legend: Legend, title: str):
        legend.set_title(title)
        legend.set_loc("best")

    def _get_columns_to_groupby(self, *, frame: pd.DataFrame) -> list[str]:
        columns_to_groupby = list[str]()
        if "estimator" in frame.columns:
            columns_to_groupby.append("estimator")
        if "label" in frame.columns:
            columns_to_groupby.append("label")
        if "output" in frame.columns:
            columns_to_groupby.append("output")
        return columns_to_groupby

    def _plot_homogeneous_estimator(
        self,
        *,
        plot_function: Callable,
        plot_function_kwargs: dict,
        subplots_by: Literal["estimator", "label", "output"] | None = None,
    ):
        frame, name = self.frame(), self.coefficients["estimator"].unique()[0]

        # {"label"} or {"output"} or {}
        columns_to_groupby = self._get_columns_to_groupby(frame=frame)
        if subplots_by is not None and not len(columns_to_groupby):
            raise ValueError("No columns to group by.")
        elif subplots_by is not None and subplots_by not in columns_to_groupby:
            raise ValueError(
                f"Column {subplots_by} not found in the frame. It should be one "
                f"of {', '.join(columns_to_groupby)}."
            )

        if subplots_by is None:
            hue = None if not len(columns_to_groupby) else columns_to_groupby[0]
            palette = "tab10" if hue is not None else None
            ncols, sharex, figsize = 1, False, (6.4, 4.8)
            self.figure_, self.ax_ = plt.subplots(
                ncols=ncols, sharex=sharex, figsize=figsize
            )
            plot_function(
                data=frame,
                x="coefficients",
                y="feature",
                hue=hue,
                palette=palette,
                ax=self.ax_,
                **plot_function_kwargs,
            )
            self._decorate_matplotlib_axis(ax=self.ax_)
            self.ax_.set_title(f"{name}")
            if hue is not None:
                self._set_legend(legend=self.ax_.get_legend(), title=hue.capitalize())
        else:
            ncols, sharex, sharey = frame[subplots_by].nunique(), True, True
            figsize = (6.4 * ncols, 4.8)
            self.figure_, self.ax_ = plt.subplots(
                ncols=ncols, sharex=sharex, sharey=sharey, figsize=figsize
            )

            for ax, (group, group_frame) in zip(
                self.ax_.flatten(), frame.groupby(by=subplots_by), strict=True
            ):
                plot_function(
                    data=group_frame,
                    x="coefficients",
                    y="feature",
                    ax=ax,
                    **plot_function_kwargs,
                )
                self._decorate_matplotlib_axis(ax=ax)
                ax.set_title(f"{name} - {subplots_by.capitalize()}: {group}")

    def _plot_estimator_report(
        self, *, subplots_by: Literal["estimator", "label", "output"] | None = None
    ):
        self._plot_homogeneous_estimator(
            subplots_by=subplots_by,
            plot_function=sns.barplot,
            plot_function_kwargs={},
        )

    def _plot_cross_validation_report(
        self, *, subplots_by: Literal["estimator", "label", "output"] | None = None
    ):
        self._plot_homogeneous_estimator(
            subplots_by=subplots_by,
            plot_function=sns.boxplot,
            plot_function_kwargs={
                "vert": False,
                "whis": 100_000,
                **BOXPLOT_STYLE,
            },
        )

    @staticmethod
    def _has_same_features(*, frame: pd.DataFrame) -> bool:
        grouped = {
            name: group["feature"].sort_values().tolist()
            for name, group in frame.groupby("estimator")
        }
        _, reference_features = grouped.popitem()
        for group_features in grouped.values():
            if group_features != reference_features:
                return False
        return True

    def _plot_heterogeneous_estimator(
        self,
        *,
        plot_function: Callable,
        plot_function_kwargs: dict,
        subplots_by: Literal["estimator", "label", "output"] | None = None,
    ):
        frame = self.frame()
        # help mypy to understand the following variable types
        hue: str | None = None
        palette: str | None = None

        # {"estimator", "label"} or {"estimator", "output"} or {"estimator"}
        columns_to_groupby = self._get_columns_to_groupby(frame=frame)
        if subplots_by is not None and subplots_by not in columns_to_groupby:
            raise ValueError(
                f"Column {subplots_by} not found in the frame. It should be one "
                f"of {', '.join(columns_to_groupby)}."
            )

        has_same_features = self._has_same_features(frame=frame)
        if (frame.columns.isin(["label", "output"]).any() and subplots_by is None) or (
            not has_same_features and subplots_by is None
        ):
            # default fallback on subplots by estimator
            # case 1: multiclass classification or multi-output regression
            # therefore, too many information to display on a single plot, by default
            # group by estimator
            # case 2: features cannot be compared across estimators and we therefore
            # need to subplots by estimator
            subplots_by = "estimator"

        if subplots_by is None:
            hue, palette = columns_to_groupby[0], "tab10"
            ncols, sharex, figsize = 1, False, (6.4, 4.8)
            self.figure_, self.ax_ = plt.subplots(
                ncols=ncols, sharex=sharex, figsize=figsize
            )
            plot_function(
                data=frame,
                x="coefficients",
                y="feature",
                hue=hue,
                palette=palette,
                ax=self.ax_,
                **plot_function_kwargs,
            )
            self._decorate_matplotlib_axis(ax=self.ax_)
            if hue is not None:
                self._set_legend(legend=self.ax_.get_legend(), title=hue.capitalize())
        else:
            ncols, sharex, sharey = (
                frame[subplots_by].nunique(),
                True,
                has_same_features,
            )
            figsize = (6.4 * ncols, 4.8)
            self.figure_, self.ax_ = plt.subplots(
                ncols=ncols, sharex=sharex, sharey=sharey, figsize=figsize
            )

            # infer if we should group by another column using hue
            hue_groupby = [col for col in columns_to_groupby if col != subplots_by]
            hue = hue_groupby[0] if len(hue_groupby) else None
            palette = "tab10" if hue is not None else None
            if not has_same_features and hue == "estimator":
                raise ValueError(
                    "The estimators have different features and should be plotted on "
                    "different axis using `subplots_by='estimator'`."
                )

            for ax, (group, group_frame) in zip(
                self.ax_.flatten(), frame.groupby(by=subplots_by), strict=True
            ):
                plot_function(
                    data=group_frame,
                    x="coefficients",
                    y="feature",
                    hue=hue,
                    palette=palette,
                    ax=ax,
                    **plot_function_kwargs,
                )
                self._decorate_matplotlib_axis(ax=ax)
                if hue is not None:
                    self._set_legend(legend=ax.get_legend(), title=hue.capitalize())
                ax.set_title(f"{subplots_by.capitalize()}: {group}")

    def _plot_comparison_report_estimator(
        self, *, subplots_by: Literal["estimator", "label", "output"] | None = None
    ):
        self._plot_heterogeneous_estimator(
            subplots_by=subplots_by,
            plot_function=sns.barplot,
            plot_function_kwargs={},
        )

    def _plot_comparison_report_cross_validation(
        self, *, subplots_by: Literal["estimator", "label", "output"] | None = None
    ):
        self._plot_heterogeneous_estimator(
            subplots_by=subplots_by,
            plot_function=sns.boxplot,
            plot_function_kwargs={
                "vert": False,
                "whis": 100_000,
                **BOXPLOT_STYLE,
            },
        )

    @classmethod
    def _compute_data_for_display(
        cls,
        *,
        estimators: Sequence[BaseEstimator],
        names: list[str],
        splits: list[int | float],
        report_type: ReportType,
    ):
        feature_names, est_names, coefficients, split_indices = [], [], [], []
        for estimator, name, split in zip(estimators, names, splits, strict=True):
            if isinstance(estimator, Pipeline):
                preprocessor, predictor = estimator[:-1], estimator[-1]
            else:
                preprocessor, predictor = None, estimator

            if isinstance(predictor, TransformedTargetRegressor):
                predictor = predictor.regressor_

            coef = np.atleast_2d(predictor.coef_).T
            intercept = np.atleast_2d(predictor.intercept_)
            coefficients.append(np.concatenate([intercept, coef]))

            feat_names = ["Intercept"] + _get_feature_names(
                predictor, transformer=preprocessor, n_features=coef.shape[0]
            )
            feature_names.extend(feat_names)
            est_names.extend([name] * len(feat_names))
            split_indices.extend([split] * len(feat_names))

        index = pd.DataFrame(
            {
                "estimator": est_names,
                "split": split_indices,
                "feature": feature_names,
            }
        )

        if coef.shape[1] == 1:
            # binary or single output regression
            columns, require_melting = ["coefficients"], False
            index["label"], index["output"] = np.nan, np.nan
        else:
            require_melting = True
            if is_classifier(predictor):
                # multi-class classification
                columns, var_name = predictor.classes_.tolist(), "label"
                index["output"] = np.nan
            else:
                # multi-output regression
                columns, var_name = [f"{i}" for i in range(coef.shape[1])], "output"
                index["label"] = np.nan
            id_vars, value_name = index.columns.tolist(), "coefficients"

        coefficients = pd.DataFrame(
            np.concatenate(coefficients, axis=0), columns=columns
        )
        coefficients = pd.concat([index, coefficients], axis=1)
        if require_melting:
            # melt the coefficients and ensure alignment with the label/output, split
            # feature names, and estimator names
            coefficients = coefficients.melt(
                id_vars=id_vars,
                value_vars=columns,
                var_name=var_name,
                value_name=value_name,
            )

        return cls(coefficients=coefficients, report_type=report_type)
