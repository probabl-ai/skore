from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve

from skore._sklearn._base import _get_cached_response_values
from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import _despine_matplotlib_axis
from skore._sklearn.types import DataSource, ReportType
from skore._utils._cache import Cache


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
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import EstimatorReport, train_test_split
    >>> X, y = make_classification(
    ...     n_samples=100_000, n_features=20, n_informative=2, n_redundant=10,
    ...     random_state=42)
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
        strategy: Literal["uniform", "quantile"] = "quantile",
        pos_label: int | None = None,
    ) -> CalibrationDisplay:
        """Compute the data for the calibration curve display."""
        cache = Cache()
        # Randomly generate estimator hash
        rng = np.random.default_rng(time.time_ns())
        estimator_hash = rng.integers(
            low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max
        )

        # Get predicted probabilities
        y_pred_cache = _get_cached_response_values(
            cache=cache,
            estimator_hash=int(estimator_hash),
            estimator=estimator,
            X=X,
            response_method="predict_proba",
            pos_label=pos_label,
        )
        # unpack actual predictions value
        pred_vals = y_pred_cache[0][1]

        # ensure ndarray-compatible
        y_pred = pred_vals if isinstance(pred_vals, np.ndarray) else pred_vals.values

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
        return cls(calibration_report=df, report_type=report_type)

    @DisplayMixin.style_plot
    def plot(self) -> None:
        """Plot the mean decrease in impurity for the different features.

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
        >>> report = EstimatorReport(RandomForestClassifier(), **split_data)
        >>> display = report.inspection.calibration_curve()
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

        return self._plot_calibration_curve_single_estimator(
            frame=self.frame(),
            estimator_name=self.calibration_report["estimator"][0],
            ref_line=True,
            hue=None,
            lineplot_kwargs=lineplot_kwargs,
        )

    def _plot_calibration_curve_single_estimator(
        self,
        *,
        frame: pd.DataFrame,
        estimator_name: str,
        hue: str | None = None,
        ref_line: bool = True,
        lineplot_kwargs: dict[str, Any],
    ):
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

        self.ax_.set_xlabel("Mean predicted probability")
        self.ax_.set_ylabel("Fraction of positives")
        self.ax_.set_xlim(0, 1)
        self.ax_.set_ylim(0, 1)

        _despine_matplotlib_axis(
            self.ax_,
            axis_to_despine=("top", "right"),
            remove_ticks=True,
            x_range=None,
            y_range=None,
        )
        self.figure_.suptitle(f"Calibration Curve of {estimator_name}")

        return self
