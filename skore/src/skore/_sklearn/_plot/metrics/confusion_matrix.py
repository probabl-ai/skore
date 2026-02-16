from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.utils._response import _check_response_method

from skore._externals._sklearn_compat import confusion_matrix_at_thresholds
from skore._sklearn._base import BaseEstimator
from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import _ClassifierDisplayMixin, _validate_style_kwargs
from skore._sklearn.types import (
    DataSource,
    MLTask,
    PositiveLabel,
    ReportType,
    YPlotData,
)

ThresholdValue = float | Literal["default"]


class ConfusionMatrixDisplay(_ClassifierDisplayMixin, DisplayMixin):
    """Display for confusion matrix.

    Parameters
    ----------
    confusion_matrix : pd.DataFrame
        Confusion matrix data in long format with columns: "True label",
        "Predicted label", "count", "normalized_by_true", "normalized_by_pred",
        "normalized_by_all" and "threshold". Each row represents one cell of one
        confusion matrix.

    display_labels : list of str
        Display labels for plot axes.

    report_type : {"comparison-cross-validation", "comparison-estimator", \
            "cross-validation", "estimator"}
        The type of report.

    ml_task : {"binary-classification", "multiclass-classification"}
        The machine learning task.

    data_source : {"test", "train", "X_y"}
        The data source to use.

    pos_label : int, float, bool, str or None
        The class considered as the positive class when displaying the confusion
        matrix.

    response_method : str
        The estimator's method that was used to get the predictions. The possible
        values are: "predict", "predict_proba", and "decision_function".

    Attributes
    ----------
    thresholds : ndarray of shape (n_thresholds,)
        Thresholds of the decision function. Each threshold is associated with a
        confusion matrix. Only available for binary classification. Thresholds are
        sorted in ascending order.

    facet_ : seaborn FacetGrid
        FacetGrid containing the confusion matrix.

    figure_ : matplotlib Figure
        Figure containing the confusion matrix.

    ax_ : matplotlib Axes
        Axes with confusion matrix.
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
        confusion_matrix: pd.DataFrame,
        display_labels: list[str],
        report_type: ReportType,
        ml_task: MLTask,
        thresholds: NDArray,
        data_source: DataSource,
        pos_label: PositiveLabel,
        response_method: str,
    ):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels
        self.report_type = report_type
        self.thresholds = thresholds
        self.ml_task = ml_task
        self.data_source = data_source
        self.pos_label = pos_label
        self.response_method = response_method

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        threshold_value: ThresholdValue = "default",
        subplot_by: Literal["split", "estimator", "auto"] | None = "auto",
    ):
        """Plot the confusion matrix.

        In binary classification, the confusion matrix can be displayed at various
        decision thresholds. This is useful for understanding how the model's
        predictions change as the decision threshold varies. Use
        ``threshold_value="default"`` to plot at the default threshold (0.5 for
        `predict_proba` response method, 0 for `decision_function` response method).

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        threshold_value : float or "default", default="default"
            The decision threshold to use when applicable (binary classification
            only). If ``"default"``, plots at the default threshold (0.5 for
            `predict_proba` response method, 0 for `decision_function` response
            method). If a float, plots at the closest available threshold.

        subplot_by: Literal["split", "estimator", "auto"] | None = "auto",
            The variable to use for subplotting. If None, the confusion matrix will not
            be subplotted. If "auto", the variable will be automatically determined
            based on the report type.

        Returns
        -------
        self : ConfusionMatrixDisplay
            Configured with the confusion matrix.
        """
        return self._plot(
            normalize=normalize,
            threshold_value=threshold_value,
            subplot_by=subplot_by,
        )

    def _plot_matplotlib(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        threshold_value: ThresholdValue = "default",
        subplot_by: Literal["split", "estimator", "auto"] | None = "auto",
    ) -> None:
        """Matplotlib implementation of the `plot` method.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        threshold_value : float or "default", default="default"
            The decision threshold to use when applicable.

        subplot_by: Literal["split", "estimator", "auto"] | None = "auto",
            The variable to use for subplotting. If None, the confusion matrix will not
            be subplotted. If "auto", the variable will be automatically determined
            based on the report type.
        """
        subplot_by_validated = self._validate_subplot_by(subplot_by, self.report_type)

        if "cross-validation" in self.report_type and subplot_by_validated != "split":
            # Aggregate the data across splits and create custom annotations.
            default_fmt = ".3f" if normalize else ".1f"
            annot_fmt = self._default_heatmap_kwargs.get("fmt", default_fmt)
            frame = self.frame(normalize=normalize, threshold_value=threshold_value)
            aggregated = (
                frame.groupby(
                    ["true_label", "predicted_label", "estimator", "data_source"]
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
            frame = self.frame(normalize=normalize, threshold_value=threshold_value)
            default_fmt = ".2f" if normalize else "d"

        heatmap_kwargs_validated = _validate_style_kwargs(
            {"fmt": default_fmt, **self._default_heatmap_kwargs}, {}
        )

        facet_grid_kwargs_validated = _validate_style_kwargs(
            {"col": subplot_by_validated, **self._default_facet_grid_kwargs}, {}
        )
        self.facet_ = sns.FacetGrid(
            data=frame,
            **facet_grid_kwargs_validated,
        )
        self.figure_, self.ax_ = self.facet_.figure, self.facet_.axes.flatten()

        def plot_heatmap(data, **kwargs):
            """Plot heatmap for each facet."""
            heatmap_data = data.pivot(
                index="true_label", columns="predicted_label", values="value"
            ).reindex(index=self.display_labels, columns=self.display_labels)

            if "cross-validation" in self.report_type and "annot" in data.columns:
                annot_data = data.pivot(
                    index="true_label", columns="predicted_label", values="annot"
                ).reindex(index=self.display_labels, columns=self.display_labels)
                if "annot" in kwargs and kwargs["annot"]:
                    kwargs["annot"] = annot_data
                kwargs["fmt"] = ""

            sns.heatmap(heatmap_data, **kwargs)

        self.facet_.map_dataframe(plot_heatmap, **heatmap_kwargs_validated)

        info_data_source = (
            f"Data source: {self.data_source.capitalize()} set"
            if self.data_source in ("train", "test")
            else "Data source: external set"
            if self.data_source == "X_y"
            else None
        )

        title = "Confusion Matrix"
        if self.ml_task == "binary-classification":
            if threshold_value == "default":
                display_threshold = (
                    0.5 if self.response_method == "predict_proba" else 0
                )
            else:
                display_threshold = threshold_value
            title = f"{title}\nDecision threshold: {display_threshold:.2f}"
        self.figure_.suptitle(f"{title}\n{info_data_source}")

        for ax in self.ax_:
            ax.set(
                xlabel="Predicted label",
                ylabel="True label",
            )
            if self.ml_task == "binary-classification" and self.pos_label is not None:
                ticklabels = [
                    f"{label}*" if label == str(self.pos_label) else label
                    for label in self.display_labels
                ]

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

        if len(self.ax_) == 1:
            self.ax_ = self.ax_[0]

    def _validate_subplot_by(
        self,
        subplot_by: Literal["split", "estimator", "auto"] | None,
        report_type: ReportType,
    ) -> Literal["split", "estimator"] | None:
        """Validate the `subplot_by` parameter.

        Parameters
        ----------
        subplot_by : Literal["split", "estimator", "auto"] | None
            The variable to use for subplotting.

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        Returns
        -------
        Literal["split", "estimator"] | None
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
        y_true: Sequence[YPlotData],
        y_pred: Sequence[YPlotData],
        *,
        report_type: ReportType,
        estimators: Sequence[BaseEstimator],
        ml_task: MLTask,
        data_source: DataSource | Literal["both"],
        display_labels: list[str],
        pos_label: PositiveLabel,
        response_method: str | list[str] | tuple[str, ...],
        **kwargs,
    ) -> "ConfusionMatrixDisplay":
        """Compute the confusion matrix for display.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True labels.

        y_pred : list of array-like of shape (n_samples,)
            Decision scores when binary classification with thresholds enabled.
            Otherwise, predicted labels.

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        estimators : list of BaseEstimator
            The estimators.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"test", "train", "X_y"}
            The data source to use.

        display_labels : list of str
            Display labels for plot.

        pos_label : int, float, bool, str or None
            The class considered as the positive class when displaying the confusion
            matrix.

        response_method : str or list of str or tuple of str
            The estimator's method to be invoked to get the predictions. The possible
            values are: `predict`, `predict_proba`, `predict_log_proba`, and
            `decision_function`.

        **kwargs : dict
            Additional keyword arguments that are ignored for compatibility with
            other metrics displays. Accepts but ignores `estimators` and `data_source`.

        Returns
        -------
        display : ConfusionMatrixDisplay
            The confusion matrix display.
        """
        pos_label_validated = cls._validate_from_predictions_params(
            y_true, y_pred, ml_task=ml_task, pos_label=pos_label
        )
        if data_source == "both":
            raise NotImplementedError(
                "Displaying both data sources is not supported yet."
            )
        data_source = cast(DataSource, data_source)
        # When provided, the positive label is set in second position.
        if ml_task == "binary-classification" and pos_label_validated is not None:
            neg_label = next(
                label for label in display_labels if label != pos_label_validated
            )
            display_labels = [str(neg_label), str(pos_label_validated)]

        cm_records = []
        for y_true_i, y_pred_i in zip(y_true, y_pred, strict=False):
            if ml_task == "binary-classification":
                tns, fps, fns, tps, thresholds = confusion_matrix_at_thresholds(
                    y_true=y_true_i.y,
                    y_score=y_pred_i.y,
                    pos_label=pos_label_validated,
                )
                cms = (
                    np.column_stack([tns, fps, fns, tps]).reshape(-1, 2, 2).astype(int)
                )
            else:
                cms = sklearn_confusion_matrix(
                    y_true=y_true_i.y,
                    y_pred=y_pred_i.y,
                    normalize=None,  # we will normalize later
                    labels=display_labels,
                )[np.newaxis, ...]
                thresholds = np.array([np.nan])

            counts = cms.reshape(-1)

            row_sums = cms.sum(axis=2, keepdims=True)
            cm_true = np.zeros_like(cms, dtype=float)
            np.divide(cms, row_sums, out=cm_true, where=row_sums != 0)
            normalized_true_values = cm_true.reshape(-1)

            col_sums = cms.sum(axis=1, keepdims=True)
            cm_pred = np.zeros_like(cms, dtype=float)
            np.divide(cms, col_sums, out=cm_pred, where=col_sums != 0)
            normalized_pred_values = cm_pred.reshape(-1)

            total_sums = cms.sum(axis=(1, 2), keepdims=True)
            cm_all = np.zeros_like(cms, dtype=float)
            np.divide(cms, total_sums, out=cm_all, where=total_sums != 0)
            normalized_all_values = cm_all.reshape(-1)

            n_thresholds = len(thresholds)
            n_classes = len(display_labels)
            n_cells = n_classes * n_classes

            true_labels = np.tile(np.repeat(display_labels, n_classes), n_thresholds)
            pred_labels = np.tile(np.tile(display_labels, n_classes), n_thresholds)
            threshold_values = np.repeat(thresholds, n_cells)

            cm_records.append(
                pd.DataFrame(
                    {
                        "true_label": true_labels,
                        "predicted_label": pred_labels,
                        "count": counts,
                        "normalized_by_true": normalized_true_values,
                        "normalized_by_pred": normalized_pred_values,
                        "normalized_by_all": normalized_all_values,
                        "threshold": threshold_values,
                        "split": y_true_i.split,
                        "estimator": y_true_i.estimator_name,
                        "data_source": y_true_i.data_source,
                    }
                )
            )
        confusion_matrix = pd.concat(cm_records)
        disp = cls(
            confusion_matrix=confusion_matrix,
            display_labels=display_labels,
            report_type=report_type,
            ml_task=ml_task,
            data_source=data_source,
            pos_label=pos_label_validated,
            response_method=_check_response_method(
                estimators[0], response_method
            ).__name__,
            thresholds=np.unique(confusion_matrix["threshold"]),
        )

        return disp

    @staticmethod
    def _format_frame(
        df: pd.DataFrame, columns: list[str], normalize_col: str
    ) -> pd.DataFrame:
        return df[columns].rename(columns={normalize_col: "value"})

    def frame(
        self,
        *,
        normalize: Literal["true", "pred", "all"] | None = None,
        threshold_value: ThresholdValue | None = "default",
    ):
        """Return the confusion matrix as a long format dataframe.

        In binary classification, the confusion matrix can be returned at various
        decision thresholds. This is useful for understanding how the model's
        predictions change as the decision threshold varies. Use
        ``threshold_value="default"`` to return the confusion matrix at the default
        threshold (0.5 for `predict_proba` response method, 0 for `decision_function`
        response method). Use ``threshold_value=None`` to return all available
        thresholds without filtering.

        The matrix is returned as a long format dataframe where each line represents one
        cell of the matrix. The columns are "true_label", "predicted_label", "value",
        "threshold", "split", "estimator", "data_source" ; where "value" is one of
        {"count", "normalized_by_true", "normalized_by_pred", "normalized_by_all"},
        depending on the value of the `normalize` parameter.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, the confusion matrix will not be
            normalized.

        threshold_value : float, "default" or None, default="default"
            The decision threshold(s) to use when applicable (binary classification
            only). If ``"default"``, returns the confusion matrix at the default
            threshold (0.5 for `predict_proba` response method, 0 for
            `decision_function` response method). If ``None``, returns all available
            thresholds without filtering. If a float, returns the confusion matrix at
            the closest available threshold to the given value.

        Returns
        -------
        frame : pandas.DataFrame
            The confusion matrix as a dataframe.
        """
        normalize_col = "normalized_by_" + normalize if normalize else "count"
        if (
            threshold_value not in ("default", None)
            and self.ml_task != "binary-classification"
        ):
            raise ValueError(
                "Threshold support is only available for binary classification."
            )

        columns = [
            "true_label",
            "predicted_label",
            normalize_col,
            "threshold",
            "split",
            "estimator",
            "data_source",
        ]

        if threshold_value is None:
            # Return all thresholds (binary) or full matrix (multiclass).
            return self._format_frame(self.confusion_matrix, columns, normalize_col)
        if threshold_value == "default":
            if self.ml_task == "binary-classification":
                threshold_value = 0.5 if self.response_method == "predict_proba" else 0
            else:
                return self._format_frame(self.confusion_matrix, columns, normalize_col)

        def select_threshold_and_format(group):
            thresholds = np.sort(group["threshold"].unique())
            index_right = int(np.searchsorted(thresholds, threshold_value))
            if index_right == len(thresholds):
                index_right = index_right - 1
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

        frames = []
        if self.report_type == "comparison-cross-validation":
            for _, group in self.confusion_matrix.groupby(["split", "estimator"]):
                frames.append(select_threshold_and_format(group))
        elif self.report_type == "cross-validation":
            for _, group in self.confusion_matrix.groupby(["split"]):
                frames.append(select_threshold_and_format(group))
        elif self.report_type == "comparison-estimator":
            for _, group in self.confusion_matrix.groupby(["estimator"]):
                frames.append(select_threshold_and_format(group))
        else:
            frames.append(select_threshold_and_format(self.confusion_matrix))

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
        self : object
            The instance with a modified style.

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
