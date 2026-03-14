from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.utils._response import _get_response_values_binary

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn.types import DataSource, ReportType


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
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import EstimatorReport, train_test_split
    >>> iris = load_iris(as_frame=True)
    >>> X, y = iris.data, iris.target
    >>> y = iris.target_names[y]
    >>> split_data = train_test_split(
    ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
    ... )
    >>> report = EstimatorReport(LogisticRegression(), **split_data)
    >>> display = report.inspection.calibration_curve(n_bins=5, strategy="uniform")
    >>> display.frame()
                predicted_probability	fraction_of_positives	data_source	label
        0	        0.049821	            0.067349	            test	   1
        1	        0.293844	            0.346451	            test	   1
        2	        0.501131	            0.553564	            test	   1
        3	        0.707339	            0.726359	            test	   1
        4	        0.944889	            0.936064	            test	   1

    """

    _default_line_kwargs = {"marker": "s", "linestyle": "-", "color": "blue"}

    def __init__(self, *, calibration_report: pd.DataFrame, report_type: ReportType):
        self.calibration_report = calibration_report
        self.report_type = report_type

    def frame(self) -> pd.DataFrame:
        """Return the data used for the display as a DataFrame."""
        if self.report_type == "estimator":
            columns_to_drop = ["estimator", "split"]
        elif self.report_type == "cross-validation":
            columns_to_drop = ["estimator"]
        elif self.report_type == "comparison-estimator":
            columns_to_drop = ["split"]
        else:  # comparison-cross-validation
            columns_to_drop = []

        frame = self.calibration_report.drop(columns=columns_to_drop)
        return frame

    @classmethod
    def _compute_data_for_display(
        cls,
        data_source: DataSource,
        estimator: BaseEstimator,
        name: str,
        X: ArrayLike,
        y: ArrayLike,
        report_type: ReportType,
        n_bins: int = 5,
        strategy: Literal["uniform", "quantile"] = "uniform",
        response_method: Literal["auto", "predict_proba", "decision_function"] = "auto",
        pos_label: int | str | None = None,
    ) -> CalibrationDisplay:
        """Compute the data for the calibration curve display."""
        # Get predicted probabilities
        y_pred, pos_label = _get_response_values_binary(
            estimator,
            X,
            response_method=response_method,
            pos_label=pos_label,
        )

        # Compute the calibration curve
        fraction_of_positives, predicted_probability = calibration_curve(
            y, y_pred, n_bins=n_bins, strategy=strategy
        )
        # Create a DataFrame for the display
        df = pd.DataFrame(
            {
                "predicted_probability": predicted_probability,
                "fraction_of_positives": fraction_of_positives,
            }
        )
        df["estimator"] = name
        df["data_source"] = data_source
        df["split"] = np.nan
        df["label"] = pos_label
        return cls(calibration_report=df, report_type=report_type)

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
        lineplot_kwargs = self._default_line_kwargs.copy()

        return self._plot_calibration_curve(
            frame=self.frame(),
            estimator_name=self.calibration_report["estimator"][0],
            report_type=self.report_type,
            ref_line=True,
            hue=None,
            lineplot_kwargs=lineplot_kwargs,
        )

    def _plot_calibration_curve(
        self,
        *,
        frame: pd.DataFrame,
        estimator_name: str,
        report_type: ReportType,
        hue: str | None = None,
        ref_line: bool = True,
        lineplot_kwargs: dict[str, Any],
    ):
        info_pos_label = (
            f"(Positive class: {frame['label'].iloc[0]})"
            if frame["label"].iloc[0] is not None
            else ""
        )
        self.plot_ = sns.lineplot(
            data=frame,
            x="predicted_probability",
            y="fraction_of_positives",
            markers=True,
            hue=hue,
            label=estimator_name,
            **lineplot_kwargs,
        )
        self.figure_, self.ax_ = self.plot_.figure, self.plot_.axes
        ref_line_label = "Perfectly calibrated"
        if ref_line:
            self.ax_.plot([0, 1], [0, 1], "k:", label=ref_line_label)

        # We always have to show the legend for at least the reference line
        self.ax_.legend(loc="lower right")
        add_background_features = hue is not None

        self.ax_.set_xlabel(f"Mean predicted probability {info_pos_label}")
        self.ax_.set_ylabel(f"Fraction of positives {info_pos_label}")
        self.ax_.set_xlim(0, 1)
        self.ax_.set_ylim(0, 1)

        if add_background_features:
            self.ax_.axhspan(
                0,
                1,
                color="lightgray",
                alpha=0.4,
                zorder=0,
            )

        return self
