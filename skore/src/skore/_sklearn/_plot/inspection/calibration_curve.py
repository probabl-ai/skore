from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import (
    _check_label,
    _despine_matplotlib_axis,
    _one_hot_encode,
)
from skore._sklearn.types import _DEFAULT, DataSource, PositiveLabel, ReportType


class CalibrationDisplay(DisplayMixin):
    """Calibration curve visualization.

    Parameters
    ----------
        calibration_report : DataFrame
        Calibration report of the estimator The columns should be:
            - `estimator`
            - `data_source`
            - `split`
            - `label`
            - `predicted_probability`
            - `fraction_of_positives`

     report_type : {"estimator", "cross-validation", "comparison-estimator", \
            "comparison-cross-validation"}
        Report type from which the display is created.

    Attributes
    ----------
    facet_ : seaborn FacetGrid
        FacetGrid containing the coefficients.

    figure_ : matplotlib Figure
        Figure containing the plot.

    ax_ : ndarray of matplotlib Axes
        Array of matplotlib Axes with the different matplotlib axis.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import EstimatorReport, train_test_split
    >>> X, y = make_classification(
    ...     n_samples=100_000, n_features=20, n_informative=2, n_redundant=10,
    ...     random_state=42)
    >>> split_data = train_test_split(
    ...     X=X, y=y, random_state=0, as_dict=True,
    ... )
    >>> report = EstimatorReport(LogisticRegression(), **split_data)
    >>> display = report.inspection.calibration_curve(n_bins=5, strategy="uniform")
    >>> display.frame()
                predicted_probability	fraction_of_positives	data_source	pos_label
        0	        0.049821	            0.067349	            test	   1
        1	        0.293844	            0.346451	            test	   1
        2	        0.501131	            0.553564	            test	   1
        3	        0.707339	            0.726359	            test	   1
        4	        0.944889	            0.936064	            test	   1

    """

    _default_line_kwargs = {"marker": "s", "linestyle": "-", "color": "blue"}
    _default_ax_set_kwargs = {
        "xlabel": "Mean predicted probability",
        "ylabel": "Fraction of positives",
        "xlim": (0, 1),
        "ylim": (0, 1),
        "aspect": "equal",
    }

    def __init__(
        self,
        *,
        calibration_report: pd.DataFrame,
        report_type: ReportType,
        report_pos_label: PositiveLabel = None,
    ):
        self.calibration_report = calibration_report
        self.report_type = report_type
        self.report_pos_label = report_pos_label

    @property
    def labels(self) -> list:
        """Return the list of labels available in the calibration data."""
        return self.calibration_report["label"].unique().tolist()

    def frame(self, *, label: PositiveLabel = _DEFAULT) -> pd.DataFrame:
        """Return the data used for the display as a DataFrame.

        Parameters
        ----------
        label : int, float, bool, str or None, default=report pos_label
            The class whose curve to select. Use ``None`` to show all classes.

        Returns
        -------
        DataFrame
        """
        label = _check_label(self.labels, label, self.report_pos_label)

        if self.report_type == "estimator":
            columns_to_drop = ["estimator", "split"]
        elif self.report_type == "cross-validation":
            columns_to_drop = ["estimator"]
        elif self.report_type == "comparison-estimator":
            columns_to_drop = ["split"]
        else:  # comparison-cross-validation
            columns_to_drop = []

        df = self.calibration_report.drop(columns=columns_to_drop)

        if label is not None:
            df = df.query("label == @label").reset_index(drop=True)
            df = df.drop(columns=["label"])

        return df

    @classmethod
    def _compute_data_for_display(
        cls,
        data_source: DataSource,
        name: str,
        y_pred: ArrayLike,
        y: ArrayLike,
        report_type: ReportType,
        n_bins: int = 5,
        strategy: Literal["uniform", "quantile"] = "quantile",
        report_pos_label: PositiveLabel = None,
    ) -> CalibrationDisplay:
        """Compute the data for the calibration curve display."""
        y_arr = np.asarray(y)
        y_pred_arr = np.asarray(y_pred)

        classes = np.unique(y_arr)
        y_true_onehot = _one_hot_encode(y_arr, classes)

        curve_dfs = []
        for class_idx, cls_label in enumerate(classes):
            frac_pos, pred_prob = calibration_curve(
                y_true_onehot[:, class_idx],
                y_pred_arr[:, class_idx],
                n_bins=n_bins,
                strategy=strategy,
            )
            df = pd.DataFrame(
                {
                    "predicted_probability": pred_prob,
                    "fraction_of_positives": frac_pos,
                }
            )
            df["estimator"] = name
            df["data_source"] = data_source
            df["label"] = cls_label
            df["split"] = np.nan
            curve_dfs.append(df)

        calibration_df = pd.concat(curve_dfs, ignore_index=True)
        return cls(
            calibration_report=calibration_df,
            report_type=report_type,
            report_pos_label=report_pos_label,
        )

    @DisplayMixin.style_plot
    def plot(self, *, label: PositiveLabel = _DEFAULT) -> Figure:
        """Plot the calibration curve.

        Parameters
        ----------
        label : int, float, bool, str or None, default=report pos_label
            The class whose curve to plot. Use ``None`` to show all classes.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import EstimatorReport, train_test_split
        >>> X, y = make_classification(
        ...     n_samples=100_000, n_features=20, n_informative=2, n_redundant=10,
        ...     random_state=42)
        >>> split_data = train_test_split(
        ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
        ... )
        >>> report = EstimatorReport(LogisticRegression(), **split_data)
        >>> display = report.inspection.calibration_curve()
        >>> display.plot()
        """
        label = _check_label(self.labels, label, self.report_pos_label)
        return self._plot(label=label)

    def _plot_matplotlib(self, *, label: PositiveLabel) -> Figure:
        """Dispatch the plotting function for matplotlib backend."""
        lineplot_kwargs = self._default_line_kwargs.copy()
        ax_set_kwargs = self._default_ax_set_kwargs.copy()
        plot_data = self.frame(label=label)
        return self._plot_calibration_curve_single_estimator(
            frame=plot_data,
            estimator_name=self.calibration_report["estimator"].iloc[0],
            lineplot_kwargs=lineplot_kwargs,
            ax_set_kwargs=ax_set_kwargs,
            label=label,
        )

    def _plot_calibration_curve_single_estimator(
        self,
        *,
        frame: pd.DataFrame,
        estimator_name: str,
        lineplot_kwargs: dict[str, Any],
        ax_set_kwargs: dict[str, Any],
        label: PositiveLabel = None,
    ):
        if "label" in frame.columns:
            # Multiple classes: let seaborn handle colours via hue
            lineplot_kwargs = {k: v for k, v in lineplot_kwargs.items() if k != "color"}
            facet = sns.lineplot(
                data=frame,
                x="predicted_probability",
                y="fraction_of_positives",
                hue="label",
                **lineplot_kwargs,
            )
        else:
            facet = sns.lineplot(
                data=frame,
                x="predicted_probability",
                y="fraction_of_positives",
                label=estimator_name,
                hue=None,
                **lineplot_kwargs,
            )

        figure = facet.figure
        ax = facet.axes
        ref_line_label = "Perfectly calibrated"
        ax.plot([0, 1], [0, 1], "k:", label=ref_line_label)

        # We always have to show the legend for at least the reference line
        ax.legend(loc="lower right")

        ax.set(**ax_set_kwargs)

        _despine_matplotlib_axis(
            ax,
            axis_to_despine=("top", "right"),
            remove_ticks=True,
            x_range=None,
            y_range=None,
        )

        title_parts = [f"Calibration Curve of {estimator_name}"]
        if label is not None:
            title_parts.append(f"Positive label: {label}")
        figure.suptitle("\n".join(title_parts))

        return figure

    # ignore the type signature because we override kwargs by specifying the name of
    # the parameters for the user.
    def set_style(  # type: ignore[override]
        self,
        *,
        policy: Literal["override", "update"] = "update",
        line_kwargs: dict | None = None,
        ax_set_kwargs: dict | None = None,
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : {"override", "update"}, default="update"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        line_kwargs : dict, default=None
            Keyword arguments passed to :func:`seaborn.lineplot`.

        ax_set_kwargs : dict, default=None
            Keyword arguments passed to :meth:`matplotlib.axes.Axes.set`.
            Useful keys include `xlabel`, `ylabel`, `xlim`, and `ylim`.

        Returns
        -------
        None
        """
        return super().set_style(
            policy=policy,
            line_kwargs=line_kwargs or {},
            ax_set_kwargs=ax_set_kwargs or {},
        )
