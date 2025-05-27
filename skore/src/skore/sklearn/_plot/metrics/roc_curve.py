from collections.abc import Sequence
from typing import Any, Literal, Optional, Union, cast

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer

from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import (
    LINESTYLE,
    HelpDisplayMixin,
    _ClassifierCurveDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
    sample_mpl_colormap,
)
from skore.sklearn.types import MLTask, PositiveLabel, ReportType, YPlotData


def _set_axis_labels(ax: Axes, info_pos_label: Union[str, None]) -> None:
    """Add axis labels."""
    xlabel = "False Positive Rate"
    ylabel = "True Positive Rate"
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


def _add_chance_level(
    ax: Axes,
    chance_level_kwargs: Union[dict, None],
    default_chance_level_kwargs: Union[dict, None],
) -> Line2D:
    """Add the chance-level line."""
    chance_level_kwargs = _validate_style_kwargs(
        {
            "label": "Chance level (AUC = 0.5)",
            "color": "k",
            "linestyle": "--",
        },
        chance_level_kwargs or default_chance_level_kwargs or {},
    )

    (chance_level,) = ax.plot((0, 1), (0, 1), **chance_level_kwargs)

    return cast(Line2D, chance_level)


class RocCurveDisplay(
    StyleDisplayMixin, HelpDisplayMixin, _ClassifierCurveDisplayMixin
):
    """ROC Curve visualization.

    An instance of this class is should created by `EstimatorReport.metrics.roc()`.
    You should not create an instance of this class directly.

    Parameters
    ----------
    roc_curve : DataFrame
        The ROC curve data to display. The columns are
        - "estimator_name"
        - "split_index" (may be null)
        - "label"
        - "threshold"
        - "fpr"
        - "tpr".

    roc_auc : DataFrame
        The ROC AUC data to display. The columns are
        - "estimator_name"
        - "split_index" (may be null)
        - "label"
        - "roc_auc".

    pos_label : int, float, bool, str or None
        The class considered as positive. Only meaningful for binary classification.

    data_source : {"train", "test", "X_y"}
        The data source used to compute the ROC curve.

    ml_task : {"binary-classification", "multiclass-classification"}
        The machine learning task.

    report_type : {"comparison-cross-validation", "comparison-estimator", \
            "cross-validation", "estimator"}
        The type of report.

    Attributes
    ----------
    ax_ : matplotlib axes or array of axes
        The axes on which the ROC curve is plotted.

    figure_ : matplotlib figure
        The figure on which the ROC curve is plotted.

    lines_ : list of matplotlib lines
        The lines of the ROC curve.

    chance_level_ : matplotlib line or list of lines or None
        The chance level line.

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
    >>> display = report.metrics.roc()
    >>> display.plot(roc_curve_kwargs={"color": "tab:red"})
    """

    _default_roc_curve_kwargs: Union[dict[str, Any], None] = None
    _default_chance_level_kwargs: Union[dict[str, Any], None] = None

    def __init__(
        self,
        *,
        roc_curve: DataFrame,
        roc_auc: DataFrame,
        pos_label: Optional[PositiveLabel],
        data_source: Literal["train", "test", "X_y"],
        ml_task: MLTask,
        report_type: ReportType,
    ) -> None:
        self.roc_curve = roc_curve
        self.roc_auc = roc_auc
        self.pos_label = pos_label
        self.data_source = data_source
        self.ml_task = ml_task
        self.report_type = report_type

    def _plot_single_estimator(
        self,
        *,
        estimator_name: str,
        roc_curve_kwargs: list[dict[str, Any]],
        plot_chance_level: bool = True,
        chance_level_kwargs: Optional[dict[str, Any]],
    ) -> tuple[Axes, list[Line2D], Union[str, None]]:
        """Plot ROC curve for a single estimator.

        Parameters
        ----------
        estimator_name : str
            The name of the estimator.

        roc_curve_kwargs : list of dict
            Additional keyword arguments to pass to matplotlib's plot function. In
            binary case, we should have a single dict. In multiclass case, we should
            have a list of dicts, one per class.

        plot_chance_level : bool, default=True
            Whether to plot the chance level.

        chance_level_kwargs : dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the chance level line.

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
        line_kwargs: dict[str, Any] = {}

        if self.ml_task == "binary-classification":
            if self.data_source in ("train", "test"):
                line_kwargs["label"] = (
                    f"{self.data_source.title()} set "
                    f"(AUC = {self.roc_auc['roc_auc'].item():0.2f})"
                )
            else:  # data_source in (None, "X_y")
                line_kwargs["label"] = f"AUC = {self.roc_auc['roc_auc'].item():0.2f}"

            line_kwargs_validated = _validate_style_kwargs(
                line_kwargs, roc_curve_kwargs[0]
            )

            (line,) = self.ax_.plot(
                self.roc_curve["fpr"],
                self.roc_curve["tpr"],
                **line_kwargs_validated,
            )
            lines.append(line)

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )

        else:  # multiclass-classification
            labels = self.roc_curve["label"].cat.categories
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"), 10 if len(labels) < 10 else len(labels)
            )

            for class_idx, class_label in enumerate(labels):
                query = f"label == {class_label}"
                roc_curve = self.roc_curve.query(query)
                roc_auc = self.roc_auc.query(query)["roc_auc"].item()

                roc_curve_kwargs_class = roc_curve_kwargs[class_idx]

                default_line_kwargs: dict[str, Any] = {"color": class_colors[class_idx]}
                if self.data_source in ("train", "test"):
                    default_line_kwargs["label"] = (
                        f"{str(class_label).title()} - {self.data_source} "
                        f"set (AUC = {roc_auc:0.2f})"
                    )
                else:  # data_source in (None, "X_y")
                    default_line_kwargs["label"] = (
                        f"{str(class_label).title()} - AUC = {roc_auc:0.2f}"
                    )

                line_kwargs = _validate_style_kwargs(
                    default_line_kwargs, roc_curve_kwargs_class
                )

                (line,) = self.ax_.plot(
                    roc_curve["fpr"],
                    roc_curve["tpr"],
                    **line_kwargs,
                )
                lines.append(line)

            info_pos_label = None  # irrelevant for multiclass

        if plot_chance_level:
            self.chance_level_ = _add_chance_level(
                self.ax_,
                chance_level_kwargs,
                self._default_chance_level_kwargs,
            )
        else:
            self.chance_level_ = None

        self.ax_.legend(bbox_to_anchor=(1.02, 1), title=estimator_name)

        return self.ax_, lines, info_pos_label

    def _plot_cross_validated_estimator(
        self,
        *,
        estimator_name: str,
        roc_curve_kwargs: list[dict[str, Any]],
        plot_chance_level: bool = True,
        chance_level_kwargs: Optional[dict[str, Any]],
    ) -> tuple[Axes, list[Line2D], Union[str, None]]:
        """Plot ROC curve for a cross-validated estimator.

        Parameters
        ----------
        estimator_name : str
            The name of the estimator.

        roc_curve_kwargs : list of dict
            List of dictionaries containing keyword arguments to customize the ROC
            curves. The length of the list should match the number of curves to plot.

        plot_chance_level : bool, default=True
            Whether to plot the chance level.

        chance_level_kwargs : dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the chance level line.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the ROC curves plotted.

        lines : list of matplotlib.lines.Line2D
            The plotted ROC curve lines.

        info_pos_label : str or None
            String containing positive label information for binary classification,
            None for multiclass.
        """
        lines: list[Line2D] = []
        line_kwargs: dict[str, Any] = {}

        if self.ml_task == "binary-classification":
            for split_idx in self.roc_curve["split_index"].cat.categories:
                query = f"label == {self.pos_label!r} & split_index == {split_idx}"
                roc_curve = self.roc_curve.query(query)
                roc_auc = self.roc_auc.query(query)["roc_auc"].item()

                line_kwargs_validated = _validate_style_kwargs(
                    line_kwargs, roc_curve_kwargs[split_idx]
                )
                line_kwargs_validated["label"] = (
                    f"Estimator of fold #{split_idx + 1} (AUC = {roc_auc:0.2f})"
                )

                (line,) = self.ax_.plot(
                    roc_curve["fpr"],
                    roc_curve["tpr"],
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
            labels = self.roc_curve["label"].cat.categories
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"), 10 if len(labels) < 10 else len(labels)
            )

            for class_idx, class_label in enumerate(labels):
                roc_auc = self.roc_auc.query(f"label == {class_label}")["roc_auc"]
                roc_curve_kwargs_class = roc_curve_kwargs[class_idx]

                for split_idx in self.roc_curve["split_index"].cat.categories:
                    roc_curve_label = self.roc_curve.query(
                        f"label == {class_label} & split_index == {split_idx}"
                    )

                    line_kwargs_validated = _validate_style_kwargs(
                        {
                            "color": class_colors[class_idx],
                            "alpha": 0.3,
                        },
                        roc_curve_kwargs_class,
                    )
                    if split_idx == 0:
                        line_kwargs_validated["label"] = (
                            f"{str(class_label).title()} "
                            f"(AUC = {roc_auc.mean():0.2f} +/- "
                            f"{roc_auc.std():0.2f})"
                        )
                    else:
                        line_kwargs_validated["label"] = None

                    (line,) = self.ax_.plot(
                        roc_curve_label["fpr"],
                        roc_curve_label["tpr"],
                        **line_kwargs_validated,
                    )
                    lines.append(line)

        if plot_chance_level:
            self.chance_level_ = _add_chance_level(
                self.ax_,
                chance_level_kwargs,
                self._default_chance_level_kwargs,
            )
        else:
            self.chance_level_ = None

        if self.data_source in ("train", "test"):
            title = f"{estimator_name} on $\\bf{{{self.data_source}}}$ set"
        else:
            title = f"{estimator_name} on $\\bf{{external}}$ set"
        self.ax_.legend(bbox_to_anchor=(1.02, 1), title=title)

        return self.ax_, lines, info_pos_label

    def _plot_comparison_estimator(
        self,
        *,
        estimator_names: list[str],
        roc_curve_kwargs: list[dict[str, Any]],
        plot_chance_level: bool = True,
        chance_level_kwargs: Optional[dict[str, Any]],
    ) -> tuple[Axes, list[Line2D], Union[str, None]]:
        """Plot ROC curve of several estimators.

        Parameters
        ----------
        estimator_names : list of str
            The names of the estimators.

        roc_curve_kwargs : list of dict
            List of dictionaries containing keyword arguments to customize the ROC
            curves. The length of the list should match the number of curves to plot.

        plot_chance_level : bool, default=True
            Whether to plot the chance level.

        chance_level_kwargs : dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the chance level line.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the ROC curves plotted.

        lines : list of matplotlib.lines.Line2D
            The plotted ROC curve lines.

        info_pos_label : str or None
            String containing positive label information for binary classification,
            None for multiclass.
        """
        lines: list[Line2D] = []
        line_kwargs: dict[str, Any] = {}

        if self.ml_task == "binary-classification":
            for est_idx, est_name in enumerate(estimator_names):
                query = f"label == {self.pos_label!r} & estimator_name == '{est_name}'"

                roc_curve = self.roc_curve.query(query)

                roc_auc = self.roc_auc.query(query)["roc_auc"].item()

                line_kwargs_validated = _validate_style_kwargs(
                    line_kwargs, roc_curve_kwargs[est_idx]
                )
                line_kwargs_validated["label"] = f"{est_name} (AUC = {roc_auc:0.2f})"
                (line,) = self.ax_.plot(
                    roc_curve["fpr"],
                    roc_curve["tpr"],
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
            labels = self.roc_curve["label"].cat.categories
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"), 10 if len(labels) < 10 else len(labels)
            )

            for est_idx, est_name in enumerate(estimator_names):
                est_color = class_colors[est_idx]

                for class_idx, class_label in enumerate(labels):
                    query = f"label == {class_label} & estimator_name == '{est_name}'"
                    roc_curve = self.roc_curve.query(query)

                    roc_auc = self.roc_auc.query(query)["roc_auc"].item()

                    class_linestyle = LINESTYLE[(class_idx % len(LINESTYLE))][1]

                    line_kwargs["color"] = est_color
                    line_kwargs["alpha"] = 0.6
                    line_kwargs["linestyle"] = class_linestyle

                    line_kwargs_validated = _validate_style_kwargs(
                        line_kwargs, roc_curve_kwargs[est_idx]
                    )
                    line_kwargs_validated["label"] = (
                        f"{est_name} - {str(class_label).title()} "
                        f"(AUC = {roc_auc:0.2f})"
                    )

                    (line,) = self.ax_.plot(
                        roc_curve["fpr"], roc_curve["tpr"], **line_kwargs_validated
                    )
                    lines.append(line)

        if plot_chance_level:
            self.chance_level_ = _add_chance_level(
                self.ax_,
                chance_level_kwargs,
                self._default_chance_level_kwargs,
            )
        else:
            self.chance_level_ = None

        self.ax_.legend(
            bbox_to_anchor=(1.02, 1),
            title=f"{self.ml_task.title()} on $\\bf{{{self.data_source}}}$ set",
        )

        return self.ax_, lines, info_pos_label

    def _plot_comparison_cross_validation(
        self,
        *,
        estimator_names: list[str],
        roc_curve_kwargs: list[dict[str, Any]],
        plot_chance_level: bool = True,
        chance_level_kwargs: Optional[dict[str, Any]],
    ) -> tuple[Axes, list[Line2D], Union[str, None]]:
        """Plot ROC curve of several cross-validations.

        Parameters
        ----------
        estimator_names : list of str
            The names of the estimators.

        roc_curve_kwargs : list of dict
            List of dictionaries containing keyword arguments to customize the ROC
            curves. The length of the list should match the number of curves to plot.

        plot_chance_level : bool, default=True
            Whether to plot the chance level.

        chance_level_kwargs : dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the chance level line.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the ROC curves plotted.

        lines : list of matplotlib.lines.Line2D
            The plotted ROC curve lines.

        info_pos_label : str or None
            String containing positive label information for binary classification,
            None for multiclass.
        """
        lines: list[Line2D] = []
        line_kwargs: dict[str, Any] = {}

        if self.ml_task == "binary-classification":
            labels = self.roc_curve["label"].cat.categories
            colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(estimator_names) < 10 else len(estimator_names),
            )
            for report_idx, estimator_name in enumerate(estimator_names):
                query = f"estimator_name == '{estimator_name}'"
                roc_curve = self.roc_curve.query(query)
                roc_auc = self.roc_auc.query(query)["roc_auc"]

                line_kwargs["color"] = colors[report_idx]
                line_kwargs["alpha"] = 0.6
                line_kwargs_validated = _validate_style_kwargs(
                    line_kwargs, roc_curve_kwargs[report_idx]
                )

                for split_index, segment in roc_curve.groupby(
                    "split_index", observed=True
                ):
                    if split_index == 0:
                        label_kwargs = {
                            "label": (
                                f"{estimator_name} "
                                f"(AUC = {roc_auc.mean():0.2f} +/- "
                                f"{roc_auc.std():0.2f})"
                            )
                        }
                    else:
                        label_kwargs = {}

                    (line,) = self.ax_.plot(
                        segment["fpr"],
                        segment["tpr"],
                        **(line_kwargs_validated | label_kwargs),
                    )
                    lines.append(line)

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )

            if plot_chance_level:
                self.chance_level_ = _add_chance_level(
                    self.ax_,
                    chance_level_kwargs,
                    self._default_chance_level_kwargs,
                )
            else:
                self.chance_level_ = None

            self.ax_.legend(
                bbox_to_anchor=(1.02, 1),
                title=f"{self.ml_task.title()} on $\\bf{{{self.data_source}}}$ set",
            )

        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            labels = self.roc_curve["label"].cat.categories
            colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(estimator_names) < 10 else len(estimator_names),
            )
            idx = 0

            for est_idx, estimator_name in enumerate(estimator_names):
                est_color = colors[est_idx]

                for label_idx, label in enumerate(labels):
                    query = f"label == {label} & estimator_name == '{estimator_name}'"
                    roc_curve = self.roc_curve.query(query)

                    roc_auc = self.roc_auc.query(query)["roc_auc"]

                    for split_index, segment in roc_curve.groupby(
                        "split_index", observed=True
                    ):
                        if split_index == 0:
                            label_kwargs = {
                                "label": (
                                    f"{estimator_name} "
                                    f"(AUC = {roc_auc.mean():0.2f} +/- "
                                    f"{roc_auc.std():0.2f})"
                                )
                            }
                        else:
                            label_kwargs = {}

                        line_kwargs["color"] = est_color
                        line_kwargs["alpha"] = 0.6
                        line_kwargs_validated = _validate_style_kwargs(
                            line_kwargs, roc_curve_kwargs[idx]
                        )

                        (line,) = self.ax_[label_idx].plot(
                            segment["fpr"],
                            segment["tpr"],
                            **(line_kwargs_validated | label_kwargs),
                        )
                        lines.append(line)

                        idx = idx + 1

                    info_pos_label = f"\n(Positive label: {label})"
                    _set_axis_labels(self.ax_[label_idx], info_pos_label)

            if plot_chance_level:
                self.chance_level_ = []
                for ax in self.ax_:
                    self.chance_level_.append(
                        _add_chance_level(
                            ax,
                            chance_level_kwargs,
                            self._default_chance_level_kwargs,
                        )
                    )
            else:
                self.chance_level_ = None

            for ax in self.ax_:
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.2),
                    title=(
                        f"{self.ml_task.title()} on $\\bf{{{self.data_source}}}$ set"
                    ),
                )

        return self.ax_, lines, info_pos_label

    @StyleDisplayMixin.style_plot
    def plot(
        self,
        *,
        estimator_name: Optional[str] = None,
        roc_curve_kwargs: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
        plot_chance_level: bool = True,
        chance_level_kwargs: Optional[dict[str, Any]] = None,
        despine: bool = True,
    ) -> None:
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        estimator_name : str, default=None
            Name of the estimator used to plot the ROC curve. If `None`, we use
            the inferred name from the estimator.

        roc_curve_kwargs : dict or list of dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the ROC curve(s).

        plot_chance_level : bool, default=True
            Whether to plot the chance level.

        chance_level_kwargs : dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the chance level line.

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

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
        >>> display = report.metrics.roc()
        >>> display.plot(roc_curve_kwargs={"color": "tab:red"})
        """
        if (
            self.report_type == "comparison-cross-validation"
            and self.ml_task == "multiclass-classification"
        ):
            n_labels = len(self.roc_auc["label"].cat.categories)
            self.figure_, self.ax_ = plt.subplots(
                ncols=n_labels, figsize=(6.4 * n_labels, 4.8)
            )
        else:
            self.figure_, self.ax_ = plt.subplots()

        if roc_curve_kwargs is None:
            roc_curve_kwargs = self._default_roc_curve_kwargs

        if self.ml_task == "binary-classification":
            n_curves = len(self.roc_auc.query(f"label == {self.pos_label!r}"))
        else:
            n_curves = len(self.roc_auc)

        roc_curve_kwargs = self._validate_curve_kwargs(
            curve_param_name="roc_curve_kwargs",
            curve_kwargs=roc_curve_kwargs,
            n_curves=n_curves,
            report_type=self.report_type,
        )

        if self.report_type == "estimator":
            self.ax_, self.lines_, info_pos_label = self._plot_single_estimator(
                estimator_name=(
                    estimator_name
                    or self.roc_auc["estimator_name"].cat.categories.item()
                ),
                roc_curve_kwargs=roc_curve_kwargs,
                plot_chance_level=plot_chance_level,
                chance_level_kwargs=chance_level_kwargs,
            )
        elif self.report_type == "cross-validation":
            self.ax_, self.lines_, info_pos_label = (
                self._plot_cross_validated_estimator(
                    estimator_name=(
                        estimator_name
                        or self.roc_auc["estimator_name"].cat.categories.item()
                    ),
                    roc_curve_kwargs=roc_curve_kwargs,
                    plot_chance_level=plot_chance_level,
                    chance_level_kwargs=chance_level_kwargs,
                )
            )
        elif self.report_type == "comparison-estimator":
            self.ax_, self.lines_, info_pos_label = self._plot_comparison_estimator(
                estimator_names=self.roc_auc["estimator_name"].cat.categories,
                roc_curve_kwargs=roc_curve_kwargs,
                plot_chance_level=plot_chance_level,
                chance_level_kwargs=chance_level_kwargs,
            )
        elif self.report_type == "comparison-cross-validation":
            self.ax_, self.lines_, info_pos_label = (
                self._plot_comparison_cross_validation(
                    estimator_names=self.roc_auc["estimator_name"].cat.categories,
                    roc_curve_kwargs=roc_curve_kwargs,
                    plot_chance_level=plot_chance_level,
                    chance_level_kwargs=chance_level_kwargs,
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
        estimator_names: list[str],
        ml_task: MLTask,
        data_source: Literal["train", "test", "X_y"],
        pos_label: Optional[PositiveLabel],
        drop_intermediate: bool = True,
    ) -> "RocCurveDisplay":
        """Private method to create a RocCurveDisplay from predictions.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True binary labels in binary classification.

        y_pred : list of ndarray of shape (n_samples,)
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions (as returned by
            "decision_function" on some classifiers).

        report_type : {"comparison-cross-validation", "comparison-estimator", \
                "cross-validation", "estimator"}
            The type of report.

        estimators : list of estimator instances
            The estimators from which `y_pred` is obtained.

        estimator_names : list of str
            Name of the estimators used to plot the ROC curve.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y"}
            The data source used to compute the ROC curve.

        pos_label : int, float, bool or str, default=None
            The class considered as the positive class when computing the
            precision and recall metrics.

        drop_intermediate : bool, default=True
            Whether to drop intermediate points with identical value.

        Returns
        -------
        display : RocCurveDisplay
            Object that stores computed values.
        """
        pos_label_validated = cls._validate_from_predictions_params(
            y_true, y_pred, ml_task=ml_task, pos_label=pos_label
        )

        roc_curve_records = []
        roc_auc_records = []

        if ml_task == "binary-classification":
            for y_true_i, y_pred_i in zip(y_true, y_pred):
                fpr_i, tpr_i, thresholds_i = roc_curve(
                    y_true_i.y,
                    y_pred_i.y,
                    pos_label=pos_label,
                    drop_intermediate=drop_intermediate,
                )
                roc_auc_i = auc(fpr_i, tpr_i)

                pos_label_validated = cast(PositiveLabel, pos_label_validated)

                for fpr, tpr, threshold in zip(fpr_i, tpr_i, thresholds_i):
                    roc_curve_records.append(
                        {
                            "estimator_name": y_true_i.estimator_name,
                            "split_index": y_true_i.split_index,
                            "label": pos_label_validated,
                            "threshold": threshold,
                            "fpr": fpr,
                            "tpr": tpr,
                        }
                    )

                roc_auc_records.append(
                    {
                        "estimator_name": y_true_i.estimator_name,
                        "split_index": y_true_i.split_index,
                        "label": pos_label_validated,
                        "roc_auc": roc_auc_i,
                    }
                )

        else:  # multiclass-classification
            # OvR fashion to collect fpr, tpr, and roc_auc
            for y_true_i, y_pred_i, est in zip(y_true, y_pred, estimators):
                label_binarizer = LabelBinarizer().fit(est.classes_)
                y_true_onehot_i: NDArray = label_binarizer.transform(y_true_i.y)
                for class_idx, class_ in enumerate(est.classes_):
                    fpr_class_i, tpr_class_i, thresholds_class_i = roc_curve(
                        y_true_onehot_i[:, class_idx],
                        y_pred_i.y[:, class_idx],
                        pos_label=None,
                        drop_intermediate=drop_intermediate,
                    )
                    roc_auc_class_i = auc(fpr_class_i, tpr_class_i)

                    for fpr, tpr, threshold in zip(
                        fpr_class_i, tpr_class_i, thresholds_class_i
                    ):
                        roc_curve_records.append(
                            {
                                "estimator_name": y_true_i.estimator_name,
                                "split_index": y_true_i.split_index,
                                "label": class_,
                                "threshold": threshold,
                                "fpr": fpr,
                                "tpr": tpr,
                            }
                        )

                    roc_auc_records.append(
                        {
                            "estimator_name": y_true_i.estimator_name,
                            "split_index": y_true_i.split_index,
                            "label": class_,
                            "roc_auc": roc_auc_class_i,
                        }
                    )

        dtypes = {
            "estimator_name": "category",
            "split_index": "category",
            "label": "category",
        }

        return cls(
            roc_curve=DataFrame.from_records(roc_curve_records).astype(dtypes),
            roc_auc=DataFrame.from_records(roc_auc_records).astype(dtypes),
            pos_label=pos_label_validated,
            data_source=data_source,
            ml_task=ml_task,
            report_type=report_type,
        )
