from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from skore._externals._sklearn_compat import confusion_matrix_at_thresholds
from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import (
    _check_label,
    _ClassifierDisplayMixin,
    _concat_frames_with_column_data,
    _one_hot_encode,
    _validate_style_kwargs,
)
from skore._sklearn.types import (
    _DEFAULT,
    DataSource,
    MLTask,
    PositiveLabel,
    ReportType,
)


class ConfusionMatrixDisplay(_ClassifierDisplayMixin, DisplayMixin):
    """Display for confusion matrix.

    Parameters
    ----------
    confusion_matrix_predict : pd.DataFrame
        Predict-based n x n confusion matrix in long format with columns:
        "true_label", "predicted_label", "count", "normalized_by_true",
        "normalized_by_pred", "normalized_by_all", "split", "estimator",
        "data_source". Always available.

    confusion_matrix_thresholded : pd.DataFrame or None
        Per-class OvR thresholded 2x2 confusion matrix in long format.
        Same columns as confusion_matrix_predict plus "threshold" and "label".
        None when the estimator only supports predict.

    report_type : {"comparison-cross-validation", "comparison-estimator", \
            "cross-validation", "estimator"}
        The type of report.

    ml_task : {"binary-classification", "multiclass-classification"}
        The machine learning task.

    data_source : {"test", "train"}
        The data source to use.

    report_pos_label : int, float, bool, str or None
        The default positive label for display.

    Attributes
    ----------
    labels : list
        Available class labels.
    """

    _default_heatmap_kwargs: dict = {
        "cmap": "Blues",
        "cbar": False,
        "annot": True,
    }

    _default_facet_grid_kwargs: dict = {
        "height": 6,
        "aspect": 1,
    }

    def __init__(
        self,
        *,
        confusion_matrix_predict: pd.DataFrame,
        confusion_matrix_thresholded: pd.DataFrame | None,
        report_type: ReportType,
        ml_task: MLTask,
        data_source: DataSource,
        report_pos_label: PositiveLabel,
    ):
        self.confusion_matrix_predict = confusion_matrix_predict
        self.confusion_matrix_thresholded = confusion_matrix_thresholded
        self.report_type = report_type
        self.ml_task = ml_task
        self.data_source = data_source
        self.report_pos_label = report_pos_label

    @property
    def labels(self):
        return self.confusion_matrix_predict["predicted_label"].unique().tolist()

    @classmethod
    def _concatenate(
        cls,
        child_displays: Sequence["ConfusionMatrixDisplay"],
        *,
        report_type: ReportType,
        column_data: dict[str, list] | None = None,
        **kwargs,  # for compatibility
    ) -> "ConfusionMatrixDisplay":
        """Build a confusion-matrix display by concatenating child displays."""
        first_display = child_displays[0]
        confusion_matrix_predict = _concat_frames_with_column_data(
            [d.confusion_matrix_predict for d in child_displays],
            column_data,
        )
        if first_display.confusion_matrix_thresholded is not None:
            confusion_matrix_thresholded = _concat_frames_with_column_data(
                [d.confusion_matrix_thresholded for d in child_displays],
                column_data,
            )
        else:
            confusion_matrix_thresholded = None

        return cls(
            confusion_matrix_predict=confusion_matrix_predict,
            confusion_matrix_thresholded=confusion_matrix_thresholded,
            report_type=report_type,
            ml_task=first_display.ml_task,
            data_source=first_display.data_source,
            report_pos_label=first_display.report_pos_label,
        )

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        threshold_value: float | None = None,
        subplot_by: Literal["split", "estimator", "auto"] | None = "auto",
        label: PositiveLabel = _DEFAULT,
    ) -> Figure:
        """Plot the confusion matrix.

        When the inspected classifier has a `predict_proba` or `decision_function`
        method, the confusion matrix can be displayed at various decision thresholds.
        This is useful for understanding how the model's predictions change as the
        decision threshold varies. In multiclass, this view is obtained by creating a
        binary problem for each label in a one-vs-rest fashion.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        threshold_value : float or None, default=None
            When None, plots the predict-based n x n confusion matrix.
            When a float, plots the thresholded 2x2 confusion matrix at the closest
            available threshold for the selected label. This is obtained in multiclass
            by creating a binary problem for the label in a one-vs-rest fashion.

        subplot_by : {"split", "estimator", "auto"} or None, default="auto"
            The variable to use for subplotting. If None, the confusion matrix will not
            be subplotted. If "auto", the variable will be automatically determined
            based on the report type.

        label : int, float, bool, str or None, default=report pos_label
            The class to consider as positive when using the thresholded view.
            Required when `threshold_value` is not None. Ignored when `threshold_value`
            is None.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the confusion matrix.
        """
        label = _check_label(self.labels, label, self.report_pos_label)
        if label is None and threshold_value is not None:
            raise ValueError(
                "Please indicate the class to consider as positive to show the "
                "thresholded confusion matrix."
            )
        return self._plot(
            normalize=normalize,
            threshold_value=threshold_value,
            subplot_by=subplot_by,
            label=label,
        )

    def _plot_matplotlib(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        threshold_value: float | None = None,
        subplot_by: Literal["split", "estimator", "auto"] | None = "auto",
        label: PositiveLabel,
    ) -> Figure:
        """Matplotlib implementation of the `plot` method."""
        subplot_by_validated = self._validate_subplot_by(subplot_by, self.report_type)

        if "cross-validation" in self.report_type and subplot_by_validated != "split":
            # Aggregate the data across splits and create custom annotations.
            default_fmt = ".3f" if normalize else ".1f"
            annot_fmt = self._default_heatmap_kwargs.get("fmt", default_fmt)
            frame = self.frame(
                normalize=normalize,
                threshold_value=threshold_value,
                label=label,
            )
            aggregated = (
                frame.groupby(
                    ["true_label", "predicted_label", "estimator", "data_source"],
                    observed=True,
                )["value"]
                .agg(["mean", "std"])
                .reset_index()
            )
            aggregated["annot"] = aggregated.apply(
                lambda row: f"{row['mean']:{annot_fmt}}\n(± {row['std']:{annot_fmt}})",
                axis=1,
            )

            frame = aggregated.rename(columns={"mean": "value"})
            default_fmt = ""
        else:
            frame = self.frame(
                normalize=normalize, threshold_value=threshold_value, label=label
            )
            default_fmt = ".2f" if normalize else "d"

        heatmap_kwargs_validated = _validate_style_kwargs(
            {"fmt": default_fmt, **self._default_heatmap_kwargs}, {}
        )

        facet_grid_kwargs_validated = _validate_style_kwargs(
            {"col": subplot_by_validated, **self._default_facet_grid_kwargs}, {}
        )
        facet = sns.FacetGrid(
            data=frame,
            **facet_grid_kwargs_validated,
        )
        figure, axes = facet.figure, facet.axes.flatten()

        display_labels = self.labels
        # The positive label is set in second position (which
        # means true-positive counts is the bottom-right cell in the matrix).
        # Usually, TP is the top-left cell, but we align with sklearn.
        if (
            self.ml_task == "multiclass-classification"
            and label is not None
            and threshold_value is not None
        ):
            display_labels = [f"not {label}", str(label)]
        elif self.ml_task == "binary-classification" and label is not None:
            display_labels = [
                next(clss for clss in display_labels if clss != label),
                label,
            ]

        def plot_heatmap(data, **kwargs):
            """Plot heatmap for each facet."""
            heatmap_data = data.pivot(
                index="true_label", columns="predicted_label", values="value"
            ).reindex(index=display_labels, columns=display_labels)
            if "cross-validation" in self.report_type and "annot" in data.columns:
                annot_data = data.pivot(
                    index="true_label", columns="predicted_label", values="annot"
                ).reindex(index=display_labels, columns=display_labels)
                if "annot" in kwargs and kwargs["annot"]:
                    kwargs["annot"] = annot_data
                kwargs["fmt"] = ""

            sns.heatmap(heatmap_data, **kwargs)

        facet.map_dataframe(plot_heatmap, **heatmap_kwargs_validated)

        info_data_source = (
            f"Data source: {self.data_source.capitalize()} set"
            if self.data_source in ("train", "test")
            else None
        )

        title = "Confusion Matrix"
        info_threshold = None
        if threshold_value is not None:
            info_threshold = f"Decision threshold: {threshold_value:.2f}"

        info_label = None
        if threshold_value is not None and label is not None:
            info_label = (
                f"Positive label: {label}"
                if self.ml_task == "binary-classification"
                else f"Label: {label}"
            )

        figure.suptitle(
            "\n".join(
                filter(None, [title, info_threshold, info_label, info_data_source])
            )
        )
        if len(axes[0].get_xticklabels()) == 2 and label is not None:
            ticklabels = [
                axes[0].get_xticklabels()[0].get_text(),
                f"{axes[0].get_xticklabels()[1].get_text()}*",
            ]
        else:
            ticklabels = None
        for ax in axes:
            ax.set(
                xlabel="Predicted label",
                ylabel="True label",
            )
            if ticklabels is not None:
                ax.set(
                    xticklabels=ticklabels,
                    yticklabels=ticklabels,
                )

                ax.text(
                    -0.15,
                    -0.15,
                    "*: the positive class",
                    fontsize=9,
                    style="italic",
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    transform=ax.transAxes,
                    bbox={
                        "boxstyle": "round",
                        "facecolor": "white",
                        "alpha": 0.8,
                        "edgecolor": "gray",
                    },
                )

        return figure

    def _validate_subplot_by(
        self,
        subplot_by: Literal["split", "estimator", "auto"] | None,
        report_type: ReportType,
    ) -> Literal["split", "estimator"] | None:
        """Validate the `subplot_by` parameter.

        Parameters
        ----------
        subplot_by : {"split", "estimator", "auto"} or None
            The variable to use for subplotting.

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        Returns
        -------
        {"split", "estimator"} or None
            The validated `subplot_by` parameter.
        """
        if subplot_by == "auto":
            if "comparison" in report_type:
                return "estimator"
            else:
                return None

        valid_subplot_by: list[Literal["split", "estimator"] | None]
        match report_type:
            case "estimator":
                valid_subplot_by = [None]
            case "cross-validation":
                valid_subplot_by = [None, "split"]
            case "comparison-estimator" | "comparison-cross-validation":
                valid_subplot_by = ["estimator"]

        if subplot_by not in valid_subplot_by:
            raise ValueError(
                f"Invalid `subplot_by` parameter. Valid options are: "
                f"{', '.join(str(s) for s in valid_subplot_by)} or auto. "
                f"Got '{subplot_by}' instead."
            )

        return subplot_by

    @classmethod
    def _compute_data_for_display(
        cls,
        y_true: NDArray | ArrayLike,
        y_pred: NDArray | ArrayLike,
        *,
        report_type: ReportType,
        estimator: BaseEstimator,
        estimator_name: str,
        ml_task: MLTask,
        data_source: DataSource | Literal["both"],
        report_pos_label: PositiveLabel | None = None,
        y_scores: NDArray | None = None,
        **kwargs,
    ) -> "ConfusionMatrixDisplay":
        """Compute the confusion matrix data for display.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels.

        y_pred : array-like of shape (n_samples,)
            Predicted labels from the estimator's predict method.

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        estimator : BaseEstimator
            The estimator.

        estimator_name : str
            The estimator name to attach to the display data.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"test", "train"}
            The data source to use.

        report_pos_label : int, float, bool, str or None
            The default positive label for display.


        y_scores : ndarray of shape (n_samples, n_classes) or None
            Probability estimates or decision function values. None when the
            estimator only supports predict.

        **kwargs : dict
            Additional keyword arguments ignored for compatibility.

        Returns
        -------
        display : ConfusionMatrixDisplay
            The confusion matrix display.
        """
        if data_source == "both":
            raise NotImplementedError(
                "Displaying both data sources is not supported yet."
            )
        data_source = cast(DataSource, data_source)

        classes = estimator.classes_

        confusion_matrix_predict = cls._build_predict_frame(
            sklearn_confusion_matrix(y_true=y_true, y_pred=y_pred),
            classes,
            estimator_name,
            data_source,
        )

        confusion_matrix_thresholded = None
        if y_scores is not None:
            y_true_onehot = _one_hot_encode(y_true, classes.tolist())
            y_scores_arr = np.asarray(y_scores)

            ovr_dfs = []
            for class_idx, label in enumerate(classes):
                ovr_df = cls._compute_data_ovr(
                    y_true=y_true_onehot[:, class_idx],
                    y_score=y_scores_arr[:, class_idx],
                    labels=[
                        next(clss for clss in classes if clss != label)
                        if ml_task == "binary-classification"
                        else f"not {label}",
                        label,
                    ],
                    label=label,
                    estimator=estimator_name,
                    data_source=data_source,
                    split=None,
                )
                ovr_dfs.append(ovr_df)

            confusion_matrix_thresholded = _concat_frames_with_column_data(ovr_dfs)

        return cls(
            confusion_matrix_predict=confusion_matrix_predict,
            confusion_matrix_thresholded=confusion_matrix_thresholded,
            report_type=report_type,
            ml_task=ml_task,
            data_source=data_source,
            report_pos_label=report_pos_label,
        )

    @staticmethod
    def _build_predict_frame(
        cm: NDArray,
        classes: list[str],
        estimator_name: str,
        data_source: DataSource,
    ) -> pd.DataFrame:
        """Build the predict-based n x n confusion matrix dataframe."""
        cm_batch = cm[np.newaxis, ...]
        n_classes = len(classes)

        counts = cm_batch.reshape(-1)

        row_sums = cm_batch.sum(axis=2, keepdims=True)
        cm_true = np.zeros_like(cm_batch, dtype=float)
        np.divide(cm_batch, row_sums, out=cm_true, where=row_sums != 0)

        col_sums = cm_batch.sum(axis=1, keepdims=True)
        cm_pred = np.zeros_like(cm_batch, dtype=float)
        np.divide(cm_batch, col_sums, out=cm_pred, where=col_sums != 0)

        total_sums = cm_batch.sum(axis=(1, 2), keepdims=True)
        cm_all = np.zeros_like(cm_batch, dtype=float)
        np.divide(cm_batch, total_sums, out=cm_all, where=total_sums != 0)

        return pd.DataFrame(
            {
                "true_label": np.repeat(classes, n_classes),
                "predicted_label": np.tile(classes, n_classes),
                "count": counts,
                "normalized_by_true": cm_true.reshape(-1),
                "normalized_by_pred": cm_pred.reshape(-1),
                "normalized_by_all": cm_all.reshape(-1),
                "split": None,
                "estimator": estimator_name,
                "data_source": data_source,
            }
        )

    @staticmethod
    def _compute_data_ovr(y_true, y_score, labels, **metadata):
        """Compute per-class OvR confusion matrix at all thresholds."""
        tns, fps, fns, tps, thresholds = confusion_matrix_at_thresholds(
            y_true=y_true, y_score=y_score, pos_label=1
        )
        cms = np.column_stack([tns, fps, fns, tps]).reshape(-1, 2, 2).astype(int)

        counts = cms.reshape(-1)

        row_sums = cms.sum(axis=2, keepdims=True)
        cm_true = np.zeros_like(cms, dtype=float)
        np.divide(cms, row_sums, out=cm_true, where=row_sums != 0)

        col_sums = cms.sum(axis=1, keepdims=True)
        cm_pred = np.zeros_like(cms, dtype=float)
        np.divide(cms, col_sums, out=cm_pred, where=col_sums != 0)

        total_sums = cms.sum(axis=(1, 2), keepdims=True)
        cm_all = np.zeros_like(cms, dtype=float)
        np.divide(cms, total_sums, out=cm_all, where=total_sums != 0)

        n_thresholds = len(thresholds)
        data = {
            "true_label": np.tile(np.repeat(labels, 2), n_thresholds),
            "predicted_label": np.tile(np.tile(labels, 2), n_thresholds),
            "count": counts,
            "normalized_by_true": cm_true.reshape(-1),
            "normalized_by_pred": cm_pred.reshape(-1),
            "normalized_by_all": cm_all.reshape(-1),
            "threshold": np.repeat(thresholds, 4),
            **metadata,
        }
        n = n_thresholds * 4
        for col in metadata:
            data[col] = pd.Series([data[col]], dtype="category").repeat(n)

        return pd.DataFrame(data)

    @staticmethod
    def _format_frame(
        df: pd.DataFrame, columns: list[str], normalize_col: str
    ) -> pd.DataFrame:
        return df[columns].rename(columns={normalize_col: "value"})

    def frame(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        threshold_value: float | Literal["all"] | None = None,
        label: PositiveLabel = _DEFAULT,
    ):
        """Return the confusion matrix as a long format dataframe.

        When the inspected classifier has a `predict_proba` or `decision_function`
        method, the confusion matrix can be displayed at various decision thresholds.
        This is useful for understanding how the model's predictions change as the
        decision threshold varies. In multiclass, this view is obtained by creating a
        binary problem for each label in a one-vs-rest fashion. Use
        `threshold_value="all"` to return all available thresholds without filtering.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        threshold_value : float, "all", or None, default=None
            When None, returns the predict-based n x n confusion matrix.
            When "all", returns the thresholded OvR data at all thresholds.
            When a float, returns the thresholded OvR data at the closest
            available threshold.

        label : int, float, bool, str or None, default=report pos_label
            The class to select when using the thresholded view. Use None to
            show all classes. Ignored when `threshold_value` is None.

        Returns
        -------
        frame : pandas.DataFrame
            The confusion matrix as a dataframe.
        """
        normalize_col = "normalized_by_" + normalize if normalize else "count"

        if threshold_value is None:
            columns = [
                "true_label",
                "predicted_label",
                normalize_col,
                "split",
                "estimator",
                "data_source",
            ]
            return self._format_frame(
                self.confusion_matrix_predict, columns, normalize_col
            )

        # Thresholded view
        if self.confusion_matrix_thresholded is None:
            raise ValueError(
                "Thresholded confusion matrices are not available. "
                "The estimator does not support predict_proba or "
                "decision_function."
            )

        label = _check_label(self.labels, label, self.report_pos_label)

        df = self.confusion_matrix_thresholded
        if label is not None:
            df = df.query("label == @label").reset_index(drop=True)

        columns = [
            "true_label",
            "predicted_label",
            normalize_col,
            "threshold",
        ]
        if label is None:
            columns.append("label")
        columns.extend(["split", "estimator", "data_source"])

        if threshold_value == "all":
            return self._format_frame(df, columns, normalize_col)

        # Snap to closest threshold per group
        def select_threshold_and_format(group):
            thresholds = np.sort(group["threshold"].unique())
            index_right = int(np.searchsorted(thresholds, threshold_value))
            if index_right == len(thresholds):
                index_right -= 1
            elif index_right == 0 and len(thresholds) > 1:
                index_right = 1
            index_left = index_right - 1
            diff_right = abs(thresholds[index_right] - threshold_value)
            diff_left = abs(thresholds[index_left] - threshold_value)
            closest_threshold_value = thresholds[
                index_right if diff_right < diff_left else index_left
            ]
            frame = group.query(f"threshold == {closest_threshold_value}")
            return self._format_frame(frame, columns, normalize_col)

        groupby_cols = []
        if "cross-validation" in self.report_type:
            groupby_cols.append("split")
        if "comparison" in self.report_type:
            groupby_cols.append("estimator")
        if label is None:
            groupby_cols.append("label")

        frames = []
        if groupby_cols:
            for _, group in df.groupby(groupby_cols, observed=True):
                frames.append(select_threshold_and_format(group))
        else:
            frames.append(select_threshold_and_format(df))

        return pd.concat(frames)

    # ignore the type signature because we override kwargs by specifying the name of
    # the parameters for the user.
    def set_style(  # type: ignore[override]
        self,
        *,
        policy: Literal["override", "update"] = "update",
        heatmap_kwargs: dict | None = None,
        facet_grid_kwargs: dict | None = None,
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : Literal["override", "update"], default="update"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        heatmap_kwargs : dict, default=None
            Additional keyword arguments to be passed to :func:`seaborn.heatmap`.

        facet_grid_kwargs : dict, default=None
            Additional keyword arguments to be passed to :class:`seaborn.FacetGrid`.

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
            heatmap_kwargs=heatmap_kwargs or {},
            facet_grid_kwargs=facet_grid_kwargs or {},
        )
