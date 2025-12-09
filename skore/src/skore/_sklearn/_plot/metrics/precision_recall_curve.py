from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import (
    LINESTYLE,
    _ClassifierCurveDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
    sample_mpl_colormap,
)
from skore._sklearn.types import (
    DataSource,
    MLTask,
    PositiveLabel,
    ReportType,
    YPlotData,
)


def _set_axis_labels(ax: Axes, info_pos_label: str | None) -> None:
    """Add axis labels."""
    xlabel = "Recall"
    ylabel = "Precision"
    if info_pos_label:
        xlabel += info_pos_label
        ylabel += info_pos_label

    ax.set(
        xlabel=xlabel,
        xlim=(-0.01, 1.01),
        ylabel=ylabel,
        ylim=(-0.01, 1.01),
        aspect="equal",
    )


MAX_N_LABELS = 5


class PrecisionRecallCurveDisplay(_ClassifierCurveDisplayMixin, DisplayMixin):
    """Precision Recall visualization.

    An instance of this class should be created by
    `EstimatorReport.metrics.precision_recall()`. You should not create an
    instance of this class directly.

    Parameters
    ----------
    precision_recall : DataFrame
        The precision-recall curve data to display. The columns are

        - `estimator_name`
        - `split` (may be null)
        - `label`
        - `threshold`
        - `precision`
        - `recall`.

    average_precision : DataFrame
        The average precision data to display. The columns are

        - `estimator_name`
        - `split` (may be null)
        - `label`
        - `average_precision`.

    pos_label : int, float, bool, str or None
        The class considered as the positive class. If None, the class will not
        be shown in the legend.

    data_source : {"train", "test", "X_y", "both"}
        The data source used to compute the precision recall curve.

    ml_task : {"binary-classification", "multiclass-classification"}
        The machine learning task.

    report_type : {"comparison-cross-validation", "comparison-estimator", \
            "cross-validation", "estimator"}
        The type of report.

    Attributes
    ----------
    ax_ : matplotlib axes or ndarray of axes
        The axes on which the precision-recall curve is plotted.

    figure_ : matplotlib figure
        The figure on which the precision-recall curve is plotted.

    lines_ : list of matplotlib lines
        The lines of the precision-recall curve.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import train_test_split
    >>> from skore import EstimatorReport
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
    >>> classifier = LogisticRegression(max_iter=10_000)
    >>> report = EstimatorReport(classifier, **split_data)
    >>> display = report.metrics.precision_recall()
    >>> display.plot(pr_curve_kwargs={"color": "tab:red"})
    """

    _default_pr_curve_kwargs: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        precision_recall: DataFrame,
        average_precision: DataFrame,
        pos_label: PositiveLabel | None,
        data_source: DataSource | Literal["both"],
        ml_task: MLTask,
        report_type: ReportType,
    ) -> None:
        self.precision_recall = precision_recall
        self.average_precision = average_precision
        self.pos_label = pos_label
        self.data_source = data_source
        self.ml_task = ml_task
        self.report_type = report_type

    def _plot_single_estimator(
        self,
        *,
        estimator_name: str,
        pr_curve_kwargs: list[dict[str, Any]],
    ) -> tuple[Axes, list[Line2D], str | None]:
        """Plot precision-recall curve for a single estimator.

        Parameters
        ----------
        estimator_name : str
            The name of the estimator.

        pr_curve_kwargs : list of dict
            Additional keyword arguments to pass to matplotlib's plot function. In
            binary case, we should have a single dict. In multiclass case, we should
            have a list of dicts, one per class.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes containing the plot.

        lines : list of matplotlib.lines.Line2D
            The plotted lines.

        info_pos_label : str or None
            String containing the positive label info for binary classification.
            None for multiclass.
        """
        lines: list[Line2D] = []
        line_kwargs: dict[str, Any] = {"drawstyle": "steps-post"}

        if self.ml_task == "binary-classification":
            line_kwargs_validated = _validate_style_kwargs(
                line_kwargs, pr_curve_kwargs[0]
            )

            def add_line_binary(
                data_source: Literal["train", "test"],
                line_kwargs: dict = line_kwargs_validated,
            ) -> None:
                precision_recall = self.precision_recall.query(
                    f"label == {self.pos_label!r} & data_source == {data_source!r}"
                )
                average_precision = self.average_precision.query(
                    f"label == {self.pos_label!r} & data_source == {data_source!r}"
                )["average_precision"].item()

                label = f"{data_source.title()} set (AP = {average_precision:0.2f})"

                (line,) = self.ax_.plot(
                    precision_recall["recall"],
                    precision_recall["precision"],
                    **(line_kwargs | {"label": label}),
                )
                lines.append(line)

            if self.data_source in ("train", "test"):
                # NOTE: Seriously, mypy?
                add_line_binary(
                    data_source=cast(Literal["train", "test"], self.data_source)
                )
            elif self.data_source == "both":
                add_line_binary(data_source="train")
                add_line_binary(data_source="test")
            else:  # if self.data_source in (None, "X_y")
                precision_recall = self.precision_recall.query(
                    f"label == {self.pos_label!r}"
                )
                average_precision = self.average_precision.query(
                    f"label == {self.pos_label!r}"
                )["average_precision"].item()

                (line,) = self.ax_.plot(
                    precision_recall["recall"],
                    precision_recall["precision"],
                    **(
                        line_kwargs_validated
                        | {"label": f"AP = {average_precision:0.2f}"}
                    ),
                )
                lines.append(line)

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
            legend_title = None

        else:  # multiclass-classification
            labels = self.precision_recall["label"].cat.categories
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(labels) < 10 else len(labels),
            )

            def add_line_multiclass(
                class_idx: int,
                class_label: Any,
                data_source: DataSource | None,
                linestyle: str = "solid",
            ) -> None:
                if data_source is None:
                    query = f"label == {class_label}"
                else:
                    query = f"label == {class_label} & data_source == {data_source!r}"

                precision_recall = self.precision_recall.query(query)
                average_precision = (
                    self.average_precision.query(query)["average_precision"]
                    .squeeze()
                    .item()
                )

                if self.data_source == "both" and data_source is not None:
                    label = (
                        f"{data_source.title()} set - "
                        f"{str(class_label).title()} "
                        f"(AP = {average_precision:0.2f})"
                    )
                else:
                    label = (
                        f"{str(class_label).title()} (AP = {average_precision:0.2f})"
                    )

                line_kwargs_validated = _validate_style_kwargs(
                    default_style_kwargs={
                        "color": class_colors[class_idx],
                        "label": label,
                        "linestyle": linestyle,
                        "drawstyle": "steps-post",
                    },
                    user_style_kwargs=pr_curve_kwargs[class_idx],
                )

                (line,) = self.ax_.plot(
                    precision_recall["recall"],
                    precision_recall["precision"],
                    **line_kwargs_validated,
                )
                lines.append(line)

            if self.data_source == "both":
                for class_idx, class_label in enumerate(labels):
                    add_line_multiclass(
                        class_idx=class_idx,
                        class_label=class_label,
                        data_source="train",
                        linestyle="dashed",
                    )
                    add_line_multiclass(
                        class_idx=class_idx,
                        class_label=class_label,
                        data_source="test",
                        linestyle="solid",
                    )
                legend_title = None
            else:
                for class_idx, class_label in enumerate(labels):
                    add_line_multiclass(
                        class_idx=class_idx,
                        class_label=class_label,
                        data_source=self.data_source,
                    )

                if self.data_source in ("train", "test"):
                    legend_title = f"{self.data_source.capitalize()} set"
                else:
                    legend_title = None
            info_pos_label = None  # irrelevant for multiclass

        _, labels = self.ax_.get_legend_handles_labels()
        if len(labels) > MAX_N_LABELS:  # too many lines to fit legend in the plot
            self.ax_.legend(bbox_to_anchor=(1.02, 1), title=legend_title)
        else:
            self.ax_.legend(loc="lower left", title=legend_title)
        self.ax_.set_title(f"Precision-Recall Curve for {estimator_name}")

        return self.ax_, lines, info_pos_label

    def _plot_cross_validated_estimator(
        self,
        *,
        estimator_name: str,
        pr_curve_kwargs: list[dict[str, Any]],
    ) -> tuple[Axes, list[Line2D], str | None]:
        """Plot precision-recall curve for a cross-validated estimator.

        Parameters
        ----------
        estimator_name : str
            The name of the estimator.

        pr_curve_kwargs : list of dict
            List of dictionaries containing keyword arguments to customize the
            precision-recall curves. The length of the list should match the number of
            curves to plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the precision-recall curves plotted.

        lines : list of matplotlib.lines.Line2D
            The plotted precision-recall curve lines.

        info_pos_label : str or None
            String containing positive label information for binary classification,
            None for multiclass.
        """
        lines: list[Line2D] = []
        line_kwargs: dict[str, Any] = {"drawstyle": "steps-post"}

        if self.ml_task == "binary-classification":
            for split_idx in self.precision_recall["split"].cat.categories:
                query = f"label == {self.pos_label!r} & split == {split_idx}"
                precision_recall = self.precision_recall.query(query)
                average_precision = self.average_precision.query(query)[
                    "average_precision"
                ].item()

                line_kwargs_validated = _validate_style_kwargs(
                    line_kwargs, pr_curve_kwargs[split_idx]
                )
                line_kwargs_validated["label"] = (
                    f"Split #{split_idx + 1} (AP = {average_precision:0.2f})"
                )

                (line,) = self.ax_.plot(
                    precision_recall["recall"],
                    precision_recall["precision"],
                    **line_kwargs_validated,
                )
                lines.append(line)

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            labels = self.precision_recall["label"].cat.categories
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(labels) < 10 else len(labels),
            )

            for class_idx, class_label in enumerate(labels):
                # precision_class = self.precision[class_]
                # recall_class = self.recall[class_]
                # average_precision_class = self.average_precision[class_]
                pr_curve_kwargs_class = pr_curve_kwargs[class_idx]

                for split_idx in self.precision_recall["split"].cat.categories:
                    query = f"label == {class_label!r} & split == {split_idx}"
                    precision_recall = self.precision_recall.query(query)
                    average_precision = self.average_precision.query(query)[
                        "average_precision"
                    ]
                    average_precision_mean = np.mean(average_precision)
                    average_precision_std = np.std(average_precision, ddof=1)

                    line_kwargs["color"] = class_colors[class_idx]
                    line_kwargs["alpha"] = 0.3
                    line_kwargs_validated = _validate_style_kwargs(
                        line_kwargs, pr_curve_kwargs_class
                    )
                    if split_idx == 0:
                        line_kwargs_validated["label"] = (
                            f"{str(class_label).title()} "
                            f"(AP = {average_precision_mean:0.2f} +/- "
                            f"{average_precision_std:0.2f})"
                        )
                    else:
                        line_kwargs_validated["label"] = None

                    (line,) = self.ax_.plot(
                        precision_recall["recall"],
                        precision_recall["precision"],
                        **line_kwargs_validated,
                    )
                    lines.append(line)

        if self.data_source in ("train", "test"):
            legend_title = f"{self.data_source.capitalize()} set"
        else:
            legend_title = "External set"
        _, labels = self.ax_.get_legend_handles_labels()
        if len(labels) > MAX_N_LABELS:  # too many lines to fit legend in the plot
            self.ax_.legend(bbox_to_anchor=(1.02, 1), title=legend_title)
        else:
            self.ax_.legend(loc="lower left", title=legend_title)
        self.ax_.set_title(f"Precision-Recall Curve for {estimator_name}")

        return self.ax_, lines, info_pos_label

    def _plot_comparison_estimator(
        self,
        *,
        estimator_names: list[str],
        pr_curve_kwargs: list[dict[str, Any]],
    ) -> tuple[Axes, list[Line2D], str | None]:
        """Plot precision-recall curve of several estimators.

        Parameters
        ----------
        estimator_names : list of str
            The names of the estimators.

        pr_curve_kwargs : list of dict
            List of dictionaries containing keyword arguments to customize the
            precision-recall curves. The length of the list should match the number of
            curves to plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the precision-recall curves plotted.

        lines : list of matplotlib.lines.Line2D
            The plotted precision-recall curve lines.

        info_pos_label : str or None
            String containing positive label information for binary classification,
            None for multiclass.
        """
        lines: list[Line2D] = []
        line_kwargs: dict[str, Any] = {"drawstyle": "steps-post"}

        if self.ml_task == "binary-classification":
            for est_idx, est_name in enumerate(estimator_names):
                query = f"label == {self.pos_label!r} & estimator_name == '{est_name}'"
                precision_recall = self.precision_recall.query(query)
                average_precision = self.average_precision.query(query)[
                    "average_precision"
                ].item()

                line_kwargs_validated = _validate_style_kwargs(
                    line_kwargs, pr_curve_kwargs[est_idx]
                )
                line_kwargs_validated["label"] = (
                    f"{est_name} (AP = {average_precision:0.2f})"
                )
                (line,) = self.ax_.plot(
                    precision_recall["recall"],
                    precision_recall["precision"],
                    **line_kwargs_validated,
                )
                lines.append(line)

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            labels = self.precision_recall["label"].cat.categories
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(labels) < 10 else len(labels),
            )

            for est_idx, est_name in enumerate(estimator_names):
                est_color = class_colors[est_idx]

                for class_idx, class_label in enumerate(labels):
                    query = f"label == {class_label!r} & estimator_name == '{est_name}'"
                    precision_recall = self.precision_recall.query(query)
                    average_precision = self.average_precision.query(query)[
                        "average_precision"
                    ].item()

                    class_linestyle = LINESTYLE[(class_idx % len(LINESTYLE))][1]
                    line_kwargs["color"] = est_color
                    line_kwargs["alpha"] = 0.6
                    line_kwargs["linestyle"] = class_linestyle
                    line_kwargs_validated = _validate_style_kwargs(
                        line_kwargs, pr_curve_kwargs[est_idx]
                    )
                    line_kwargs_validated["label"] = (
                        f"{est_name} - {str(class_label).title()} "
                        f"(AP = {average_precision:0.2f})"
                    )

                    (line,) = self.ax_.plot(
                        precision_recall["recall"],
                        precision_recall["precision"],
                        **line_kwargs_validated,
                    )
                    lines.append(line)

        if self.data_source in ("train", "test"):
            legend_title = f"{self.data_source.capitalize()} set"
        else:
            legend_title = "External set"

        _, labels = self.ax_.get_legend_handles_labels()
        if len(labels) > MAX_N_LABELS:  # too many lines to fit legend in the plot
            self.ax_.legend(bbox_to_anchor=(1.02, 1), title=legend_title)
        else:
            self.ax_.legend(loc="lower left", title=legend_title)
        self.ax_.set_title("Precision-Recall Curve")

        return self.ax_, lines, info_pos_label

    def _plot_comparison_cross_validation(
        self,
        *,
        estimator_names: list[str],
        pr_curve_kwargs: list[dict[str, Any]],
    ) -> tuple[Axes, list[Line2D], str | None]:
        """Plot precision-recall curve of several estimators.

        Parameters
        ----------
        estimator_names : list of str
            The names of the estimators.

        pr_curve_kwargs : list of dict
            List of dictionaries containing keyword arguments to customize the
            precision-recall curves. The length of the list should match the number of
            curves to plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the precision-recall curves plotted.

        lines : list of matplotlib.lines.Line2D
            The plotted precision-recall curve lines.

        info_pos_label : str or None
            String containing positive label information for binary classification,
            None for multiclass.
        """
        lines: list[Line2D] = []
        line_kwargs: dict[str, Any] = {"drawstyle": "steps-post"}

        if self.ml_task == "binary-classification":
            labels = self.precision_recall["label"].cat.categories
            colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(estimator_names) < 10 else len(estimator_names),
            )
            curve_idx = 0

            for report_idx, estimator_name in enumerate(estimator_names):
                query = "label == @self.pos_label & estimator_name == @estimator_name"
                average_precision = self.average_precision.query(query)[
                    "average_precision"
                ]

                precision_recall = self.precision_recall.query(query)

                for split_idx, segment in precision_recall.groupby(
                    "split", observed=True
                ):
                    if split_idx == 0:
                        label_kwargs = {
                            "label": (
                                f"{estimator_name} "
                                f"(AUC = {average_precision.mean():0.2f} "
                                f"+/- {average_precision.std():0.2f})"
                            )
                        }
                    else:
                        label_kwargs = {}

                    line_kwargs["color"] = colors[report_idx]
                    line_kwargs["alpha"] = 0.6
                    line_kwargs_validated = _validate_style_kwargs(
                        line_kwargs, pr_curve_kwargs[curve_idx]
                    )

                    (line,) = self.ax_.plot(
                        segment["recall"],
                        segment["precision"],
                        **(line_kwargs_validated | label_kwargs),
                    )
                    lines.append(line)

                    curve_idx += 1

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )

            if self.data_source in ("train", "test"):
                legend_title = f"{self.data_source.capitalize()} set"
            else:
                legend_title = "External set"

            _, labels = self.ax_.get_legend_handles_labels()
            if len(labels) > MAX_N_LABELS:  # too many lines to fit legend in the plot
                self.ax_.legend(bbox_to_anchor=(1.02, 1), title=legend_title)
            else:
                self.ax_.legend(loc="lower left", title=legend_title)
            self.ax_.set_title("Precision-Recall Curve")

        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            labels = self.precision_recall["label"].cat.categories
            colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(estimator_names) < 10 else len(estimator_names),
            )
            idx = 0

            for est_idx, estimator_name in enumerate(estimator_names):
                est_color = colors[est_idx]

                for label_idx, label in enumerate(labels):
                    query = "label == @label & estimator_name == @estimator_name"
                    average_precision = self.average_precision.query(query)[
                        "average_precision"
                    ]

                    precision_recall = self.precision_recall.query(query)

                    for split_idx, segment in precision_recall.groupby(
                        "split", observed=True
                    ):
                        if split_idx == 0:
                            label_kwargs = {
                                "label": (
                                    f"{estimator_name} "
                                    f"(AUC = {average_precision.mean():0.2f} "
                                    f"+/- {average_precision.std():0.2f})"
                                )
                            }
                        else:
                            label_kwargs = {}

                        line_kwargs["color"] = est_color
                        line_kwargs["alpha"] = 0.6
                        line_kwargs_validated = _validate_style_kwargs(
                            line_kwargs, pr_curve_kwargs[idx]
                        )

                        (line,) = self.ax_[label_idx].plot(
                            segment["recall"],
                            segment["precision"],
                            **(line_kwargs_validated | label_kwargs),
                        )
                        lines.append(line)

                        idx = idx + 1

                    info_pos_label = f"\n(Positive label: {label})"
                    _set_axis_labels(self.ax_[label_idx], info_pos_label)

            if self.data_source in ("train", "test"):
                legend_title = f"{self.data_source.capitalize()} set"
            else:
                legend_title = "External set"

            for ax in self.ax_:
                _, labels = ax.get_legend_handles_labels()
                ax.legend(loc="lower left", title=legend_title)

            self.figure_.suptitle("Precision-Recall Curve")

        return self.ax_, lines, info_pos_label

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        estimator_name: str | None = None,
        pr_curve_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        despine: bool = True,
    ) -> None:
        """Plot visualization.

        Parameters
        ----------
        estimator_name : str, default=None
            Name of the estimator used to plot the precision-recall curve. If
            `None`, we use the inferred name from the estimator.

        pr_curve_kwargs : dict or list of dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the precision-recall curve(s).

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

        Notes
        -----
        The average precision (cf. :func:`~sklearn.metrics.average_precision_score`)
        in scikit-learn is computed without any interpolation. To be consistent
        with this metric, the precision-recall curve is plotted without any
        interpolation as well (step-wise style).

        You can change this style by passing the keyword argument
        `drawstyle="default"`. However, the curve will not be strictly
        consistent with the reported average precision.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> display = report.metrics.precision_recall()
        >>> display.plot(pr_curve_kwargs={"color": "tab:red"})
        """
        return self._plot(
            estimator_name=estimator_name,
            pr_curve_kwargs=pr_curve_kwargs,
            despine=despine,
        )

    def _plot_matplotlib(
        self,
        *,
        estimator_name: str | None = None,
        pr_curve_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        despine: bool = True,
    ) -> None:
        """Matplotlib implementation of the `plot` method."""
        if (
            self.report_type == "comparison-cross-validation"
            and self.ml_task == "multiclass-classification"
        ):
            n_labels = len(self.average_precision["label"].cat.categories)
            self.figure_, self.ax_ = plt.subplots(
                ncols=n_labels, figsize=(6.4 * n_labels, 4.8)
            )
        else:
            self.figure_, self.ax_ = plt.subplots()

        if pr_curve_kwargs is None:
            pr_curve_kwargs = self._default_pr_curve_kwargs

        if self.ml_task == "binary-classification":
            n_curves = len(self.average_precision.query(f"label == {self.pos_label!r}"))
        else:
            n_curves = len(self.average_precision)

        pr_curve_kwargs = self._validate_curve_kwargs(
            curve_param_name="pr_curve_kwargs",
            curve_kwargs=pr_curve_kwargs,
            n_curves=n_curves,
            report_type=self.report_type,
        )

        if self.report_type == "estimator":
            self.ax_, self.lines_, info_pos_label = self._plot_single_estimator(
                estimator_name=(
                    self.precision_recall["estimator_name"].cat.categories.item()
                    if estimator_name is None
                    else estimator_name
                ),
                pr_curve_kwargs=pr_curve_kwargs,
            )
        elif self.report_type == "cross-validation":
            self.ax_, self.lines_, info_pos_label = (
                self._plot_cross_validated_estimator(
                    estimator_name=(
                        self.precision_recall["estimator_name"].cat.categories.item()
                        if estimator_name is None
                        else estimator_name
                    ),
                    pr_curve_kwargs=pr_curve_kwargs,
                )
            )
        elif self.report_type == "comparison-estimator":
            self.ax_, self.lines_, info_pos_label = self._plot_comparison_estimator(
                estimator_names=self.precision_recall["estimator_name"].cat.categories,
                pr_curve_kwargs=pr_curve_kwargs,
            )
        elif self.report_type == "comparison-cross-validation":
            self.ax_, self.lines_, info_pos_label = (
                self._plot_comparison_cross_validation(
                    estimator_names=self.precision_recall[
                        "estimator_name"
                    ].cat.categories,
                    pr_curve_kwargs=pr_curve_kwargs,
                )
            )
        else:
            raise ValueError(
                "`report_type` should be one of 'estimator', 'cross-validation', "
                "'comparison-cross-validation' or 'comparison-estimator'. "
                f"Got '{self.report_type}' instead."
            )

        if (
            self.report_type == "comparison-cross-validation"
            and self.ml_task == "multiclass-classification"
        ):
            for ax in self.ax_:
                if despine:
                    _despine_matplotlib_axis(ax)
        else:
            _set_axis_labels(self.ax_, info_pos_label)

            if despine:
                _despine_matplotlib_axis(self.ax_)

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
        pos_label: PositiveLabel | None,
        drop_intermediate: bool = True,
    ) -> "PrecisionRecallCurveDisplay":
        """Plot precision-recall curve given binary class predictions.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True binary labels.

        y_pred : list of array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions (as returned by
            "decision_function" on some classifiers).

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        estimators : list of estimator instances
            The estimators from which `y_pred` is obtained.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y", "both"}
            The data source used to compute the precision recall curve.

        pos_label : int, float, bool, str or none
            The class considered as the positive class when computing the
            precision and recall metrics.

        drop_intermediate : bool, default=True
            Whether to drop some suboptimal thresholds which would not appear
            on a plotted precision-recall curve. This is useful in order to
            create lighter precision-recall curves.

        Returns
        -------
        display : PrecisionRecallCurveDisplay
        """
        pos_label_validated = cls._validate_from_predictions_params(
            y_true, y_pred, ml_task=ml_task, pos_label=pos_label
        )

        precision_recall_records = []
        average_precision_records = []

        if ml_task == "binary-classification":
            for y_true_i, y_pred_i in zip(y_true, y_pred, strict=False):
                pos_label_validated = cast(PositiveLabel, pos_label_validated)
                precision_i, recall_i, thresholds_i = precision_recall_curve(
                    y_true_i.y,
                    y_pred_i.y,
                    pos_label=pos_label_validated,
                    drop_intermediate=drop_intermediate,
                )
                average_precision_i = average_precision_score(
                    y_true_i.y, y_pred_i.y, pos_label=pos_label_validated
                )

                for precision, recall, threshold in zip(
                    precision_i, recall_i, thresholds_i, strict=False
                ):
                    precision_recall_records.append(
                        {
                            "estimator_name": y_true_i.estimator_name,
                            "data_source": y_true_i.data_source,
                            "split": y_true_i.split,
                            "label": pos_label_validated,
                            "threshold": threshold,
                            "precision": precision,
                            "recall": recall,
                        }
                    )
                average_precision_records.append(
                    {
                        "estimator_name": y_true_i.estimator_name,
                        "data_source": y_true_i.data_source,
                        "split": y_true_i.split,
                        "label": pos_label_validated,
                        "average_precision": average_precision_i,
                    }
                )
        else:  # multiclass-classification
            classes = estimators[0].classes_
            for y_true_i, y_pred_i in zip(y_true, y_pred, strict=True):
                label_binarizer = LabelBinarizer().fit(classes)
                y_true_onehot_i: NDArray = label_binarizer.transform(y_true_i.y)
                y_pred_i_y = cast(NDArray, y_pred_i.y)

                for class_idx, class_ in enumerate(classes):
                    precision_class_i, recall_class_i, thresholds_class_i = (
                        precision_recall_curve(
                            y_true_onehot_i[:, class_idx],
                            y_pred_i_y[:, class_idx],
                            pos_label=None,
                            drop_intermediate=drop_intermediate,
                        )
                    )
                    average_precision_class_i = average_precision_score(
                        y_true_onehot_i[:, class_idx], y_pred_i_y[:, class_idx]
                    )

                    for precision, recall, threshold in zip(
                        precision_class_i,
                        recall_class_i,
                        thresholds_class_i,
                        strict=False,
                    ):
                        precision_recall_records.append(
                            {
                                "estimator_name": y_true_i.estimator_name,
                                "data_source": y_true_i.data_source,
                                "split": y_true_i.split,
                                "label": class_,
                                "threshold": threshold,
                                "precision": precision,
                                "recall": recall,
                            }
                        )
                    average_precision_records.append(
                        {
                            "estimator_name": y_true_i.estimator_name,
                            "data_source": y_true_i.data_source,
                            "split": y_true_i.split,
                            "label": class_,
                            "average_precision": average_precision_class_i,
                        }
                    )

        dtypes = {
            "estimator_name": "category",
            "data_source": "category",
            "split": "category",
            "label": "category",
        }

        return cls(
            precision_recall=DataFrame.from_records(precision_recall_records).astype(
                dtypes
            ),
            average_precision=DataFrame.from_records(average_precision_records).astype(
                dtypes
            ),
            pos_label=pos_label_validated,
            data_source=data_source,
            ml_task=ml_task,
            report_type=report_type,
        )

    def frame(self, with_average_precision: bool = False) -> DataFrame:
        """Get the data used to create the precision-recall curve plot.

        Parameters
        ----------
        with_average_precision : bool, default=False
            Whether to include the average precision column in the returned DataFrame.

        Returns
        -------
        DataFrame
            A DataFrame containing the precision-recall curve data with columns
            depending on the report type:

            - `estimator_name`: Name of the estimator (when comparing estimators)
            - `split`: Cross-validation split ID (when doing cross-validation)
            - `label`: Class label (for multiclass-classification)
            - `threshold`: Decision threshold
            - `precision`: Precision score at threshold
            - `recall`: Recall score at threshold
            - `average_precision`: average precision
              (when `with_average_precision=True`)

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split, EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> clf = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(clf, **split_data)
        >>> display = report.metrics.precision_recall()
        >>> df = display.frame()
        """
        if with_average_precision:
            # The merge between the precision-recall curve and the average precision is
            # done without specifying the columns to merge on, hence done on all column
            # that are present in both DataFrames.
            # In this case, the common columns are all columns but not the ones
            # containing the statistics.
            df = self.precision_recall.merge(self.average_precision)
        else:
            df = self.precision_recall

        statistical_columns = ["threshold", "precision", "recall"]
        if with_average_precision:
            statistical_columns.append("average_precision")

        if self.report_type == "estimator":
            indexing_columns = []
        elif self.report_type == "cross-validation":
            indexing_columns = ["split"]
        elif self.report_type == "comparison-estimator":
            indexing_columns = ["estimator_name"]
        else:  # self.report_type == "comparison-cross-validation"
            indexing_columns = ["estimator_name", "split"]

        if self.data_source == "both":
            indexing_columns += ["data_source"]

        if self.ml_task == "binary-classification":
            columns = indexing_columns + statistical_columns
        else:
            columns = indexing_columns + ["label"] + statistical_columns

        return df[columns]
