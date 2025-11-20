from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import auc, roc_curve
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

MAX_N_LABELS = 6  # 5 + 1 for the chance level line


def _set_axis_labels(ax: Axes, info_pos_label: str | None) -> None:
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
    chance_level_kwargs: dict | None,
    default_chance_level_kwargs: dict | None,
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


class RocCurveDisplay(_ClassifierCurveDisplayMixin, DisplayMixin):
    """ROC Curve visualization.

    An instance of this class should be created by `EstimatorReport.metrics.roc()`.
    You should not create an instance of this class directly.

    Parameters
    ----------
    roc_curve : DataFrame
        The ROC curve data to display. The columns are

        - `estimator_name`
        - `split` (may be null)
        - `label`
        - `threshold`
        - `fpr`
        - `tpr`.

    roc_auc : DataFrame
        The ROC AUC data to display. The columns are

        - `estimator_name`
        - `split` (may be null)
        - `label`
        - `roc_auc`.

    pos_label : int, float, bool, str or None
        The class considered as positive. Only meaningful for binary classification.

    data_source : {"train", "test", "X_y", "both"}
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

    _default_roc_curve_kwargs: dict[str, Any] | None = None
    _default_chance_level_kwargs: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        roc_curve: DataFrame,
        roc_auc: DataFrame,
        pos_label: PositiveLabel | None,
        data_source: DataSource | Literal["both"],
        ml_task: MLTask,
        report_type: ReportType,
    ) -> None:
        self.roc_curve = roc_curve
        self.roc_auc = roc_auc
        self.pos_label = pos_label
        self.data_source = data_source
        self.ml_task = ml_task
        self.report_type = report_type

        self.chance_level_: Line2D | list[Line2D] | None

    def _plot_single_estimator(
        self,
        *,
        estimator_name: str,
        roc_curve_kwargs: list[dict[str, Any]],
        plot_chance_level: bool = True,
        chance_level_kwargs: dict[str, Any] | None,
    ) -> tuple[Axes, list[Line2D], str | None]:
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
            line_kwargs_validated = _validate_style_kwargs(
                line_kwargs, roc_curve_kwargs[0]
            )

            def add_line_binary(
                data_source: Literal["train", "test"],
                line_kwargs: dict = line_kwargs_validated,
            ) -> None:
                roc_curve = self.roc_curve.query(f"data_source == {data_source!r}")
                roc_auc = self.roc_auc.query(f"data_source == {data_source!r}")
                label = (
                    f"{data_source.title()} set "
                    f"(AUC = {roc_auc['roc_auc'].item():0.2f})"
                )

                (line,) = self.ax_.plot(
                    roc_curve["fpr"],
                    roc_curve["tpr"],
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
                (line,) = self.ax_.plot(
                    self.roc_curve["fpr"],
                    self.roc_curve["tpr"],
                    **(
                        line_kwargs_validated
                        | {"label": f"AUC = {self.roc_auc['roc_auc'].item():0.2f}"}
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
            labels = self.roc_curve["label"].cat.categories
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"), 10 if len(labels) < 10 else len(labels)
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

                roc_curve = self.roc_curve.query(query)
                roc_auc = self.roc_auc.query(query)["roc_auc"].squeeze().item()

                if self.data_source == "both" and data_source is not None:
                    label = (
                        f"{data_source.title()} set - "
                        f"{str(class_label).title()} "
                        f"(AUC = {roc_auc:0.2f})"
                    )
                else:
                    label = f"{str(class_label).title()} (AUC = {roc_auc:0.2f})"

                line_kwargs = _validate_style_kwargs(
                    default_style_kwargs={
                        "color": class_colors[class_idx],
                        "label": label,
                        "linestyle": linestyle,
                    },
                    user_style_kwargs=roc_curve_kwargs[class_idx],
                )

                (line,) = self.ax_.plot(
                    roc_curve["fpr"],
                    roc_curve["tpr"],
                    **line_kwargs,
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

        if plot_chance_level:
            self.chance_level_ = _add_chance_level(
                self.ax_,
                chance_level_kwargs,
                self._default_chance_level_kwargs,
            )
        else:
            self.chance_level_ = None

        _, labels = self.ax_.get_legend_handles_labels()
        if len(labels) > MAX_N_LABELS:  # too many lines to fit legend in the plot
            self.ax_.legend(bbox_to_anchor=(1.02, 1), title=legend_title)
        else:
            self.ax_.legend(loc="lower right", title=legend_title)
        self.ax_.set_title(f"ROC Curve for {estimator_name}")

        return self.ax_, lines, info_pos_label

    def _plot_cross_validated_estimator(
        self,
        *,
        estimator_name: str,
        roc_curve_kwargs: list[dict[str, Any]],
        plot_chance_level: bool = True,
        chance_level_kwargs: dict[str, Any] | None,
    ) -> tuple[Axes, list[Line2D], str | None]:
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
            for split_idx in self.roc_curve["split"].cat.categories:
                query = f"label == {self.pos_label!r} & split == {split_idx}"
                roc_curve = self.roc_curve.query(query)
                roc_auc = self.roc_auc.query(query)["roc_auc"].item()

                line_kwargs_validated = _validate_style_kwargs(
                    line_kwargs, roc_curve_kwargs[split_idx]
                )
                line_kwargs_validated["label"] = (
                    f"Split #{split_idx + 1} (AUC = {roc_auc:0.2f})"
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

                for split_idx in self.roc_curve["split"].cat.categories:
                    roc_curve_label = self.roc_curve.query(
                        f"label == {class_label} & split == {split_idx}"
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
            legend_title = f"{self.data_source.capitalize()} set"
        else:
            legend_title = "External set"

        _, labels = self.ax_.get_legend_handles_labels()
        if len(labels) > MAX_N_LABELS:  # too many lines to fit legend in the plot
            self.ax_.legend(bbox_to_anchor=(1.02, 1), title=legend_title)
        else:
            self.ax_.legend(loc="lower right", title=legend_title)
        self.ax_.set_title(f"ROC Curve for {estimator_name}")

        return self.ax_, lines, info_pos_label

    def _plot_comparison_estimator(
        self,
        *,
        estimator_names: list[str],
        roc_curve_kwargs: list[dict[str, Any]],
        plot_chance_level: bool = True,
        chance_level_kwargs: dict[str, Any] | None,
    ) -> tuple[Axes, list[Line2D], str | None]:
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

        if self.data_source in ("train", "test"):
            legend_title = f"{self.data_source.capitalize()} set"
        else:
            legend_title = "External set"

        _, labels = self.ax_.get_legend_handles_labels()
        if len(labels) > MAX_N_LABELS:  # too many lines to fit legend in the plot
            self.ax_.legend(bbox_to_anchor=(1.02, 1), title=legend_title)
        else:
            self.ax_.legend(loc="lower right", title=legend_title)
        self.ax_.set_title("ROC Curve")

        return self.ax_, lines, info_pos_label

    def _plot_comparison_cross_validation(
        self,
        *,
        estimator_names: list[str],
        roc_curve_kwargs: list[dict[str, Any]],
        plot_chance_level: bool = True,
        chance_level_kwargs: dict[str, Any] | None,
    ) -> tuple[Axes, list[Line2D], str | None]:
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

                for split_idx, segment in roc_curve.groupby("split", observed=True):
                    if split_idx == 0:
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

            if self.data_source in ("train", "test"):
                legend_title = f"{self.data_source.capitalize()} set"
            else:
                legend_title = "External set"

            _, labels = self.ax_.get_legend_handles_labels()
            if len(labels) > MAX_N_LABELS:  # too many lines to fit legend in the plot
                self.ax_.legend(bbox_to_anchor=(1.02, 1), title=legend_title)
            else:
                self.ax_.legend(loc="lower right", title=legend_title)
            self.ax_.set_title("ROC Curve")

        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            labels = self.roc_curve["label"].cat.categories
            colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(estimator_names) < 10 else len(estimator_names),
            )
            curve_idx = 0

            for est_idx, estimator_name in enumerate(estimator_names):
                est_color = colors[est_idx]

                for label_idx, label in enumerate(labels):
                    query = f"label == {label} & estimator_name == '{estimator_name}'"
                    roc_curve = self.roc_curve.query(query)

                    roc_auc = self.roc_auc.query(query)["roc_auc"]

                    for split_idx, segment in roc_curve.groupby("split", observed=True):
                        if split_idx == 0:
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
                            line_kwargs, roc_curve_kwargs[curve_idx]
                        )

                        (line,) = self.ax_[label_idx].plot(
                            segment["fpr"],
                            segment["tpr"],
                            **(line_kwargs_validated | label_kwargs),
                        )
                        lines.append(line)

                        curve_idx += 1

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

            if self.data_source in ("train", "test"):
                legend_title = f"{self.data_source.capitalize()} set"
            else:
                legend_title = "External set"

            for ax in self.ax_:
                _, labels = ax.get_legend_handles_labels()
                ax.legend(loc="lower right", title=legend_title)

            self.figure_.suptitle("ROC Curve")

        return self.ax_, lines, info_pos_label

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        estimator_name: str | None = None,
        roc_curve_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        plot_chance_level: bool = True,
        chance_level_kwargs: dict[str, Any] | None = None,
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
        return self._plot(
            estimator_name=estimator_name,
            roc_curve_kwargs=roc_curve_kwargs,
            plot_chance_level=plot_chance_level,
            chance_level_kwargs=chance_level_kwargs,
            despine=despine,
        )

    def _plot_matplotlib(
        self,
        *,
        estimator_name: str | None = None,
        roc_curve_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        plot_chance_level: bool = True,
        chance_level_kwargs: dict[str, Any] | None = None,
        despine: bool = True,
    ) -> None:
        """Matplotlib implementation of the `plot` method."""
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
        ml_task: MLTask,
        data_source: DataSource | Literal["both"],
        pos_label: PositiveLabel | None,
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

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y", "both"}
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
            for y_true_i, y_pred_i in zip(y_true, y_pred, strict=False):
                fpr_i, tpr_i, thresholds_i = roc_curve(
                    y_true_i.y,
                    y_pred_i.y,
                    pos_label=pos_label,
                    drop_intermediate=drop_intermediate,
                )
                roc_auc_i = auc(fpr_i, tpr_i)

                pos_label_validated = cast(PositiveLabel, pos_label_validated)

                for fpr, tpr, threshold in zip(
                    fpr_i, tpr_i, thresholds_i, strict=False
                ):
                    roc_curve_records.append(
                        {
                            "estimator_name": y_true_i.estimator_name,
                            "data_source": y_true_i.data_source,
                            "split": y_true_i.split,
                            "label": pos_label_validated,
                            "threshold": threshold,
                            "fpr": fpr,
                            "tpr": tpr,
                        }
                    )

                roc_auc_records.append(
                    {
                        "estimator_name": y_true_i.estimator_name,
                        "data_source": y_true_i.data_source,
                        "split": y_true_i.split,
                        "label": pos_label_validated,
                        "roc_auc": roc_auc_i,
                    }
                )

        else:  # multiclass-classification
            classes = estimators[0].classes_
            # OvR fashion to collect fpr, tpr, and roc_auc
            for y_true_i, y_pred_i in zip(y_true, y_pred, strict=True):
                label_binarizer = LabelBinarizer().fit(classes)
                y_true_onehot_i: NDArray = label_binarizer.transform(y_true_i.y)
                y_pred_i_y = cast(NDArray, y_pred_i.y)

                for class_idx, class_ in enumerate(classes):
                    fpr_class_i, tpr_class_i, thresholds_class_i = roc_curve(
                        y_true_onehot_i[:, class_idx],
                        y_pred_i_y[:, class_idx],
                        pos_label=None,
                        drop_intermediate=drop_intermediate,
                    )
                    roc_auc_class_i = auc(fpr_class_i, tpr_class_i)

                    for fpr, tpr, threshold in zip(
                        fpr_class_i, tpr_class_i, thresholds_class_i, strict=False
                    ):
                        roc_curve_records.append(
                            {
                                "estimator_name": y_true_i.estimator_name,
                                "data_source": y_true_i.data_source,
                                "split": y_true_i.split,
                                "label": class_,
                                "threshold": threshold,
                                "fpr": fpr,
                                "tpr": tpr,
                            }
                        )

                    roc_auc_records.append(
                        {
                            "estimator_name": y_true_i.estimator_name,
                            "data_source": y_true_i.data_source,
                            "split": y_true_i.split,
                            "label": class_,
                            "roc_auc": roc_auc_class_i,
                        }
                    )

        dtypes = {
            "estimator_name": "category",
            "data_source": "category",
            "split": "category",
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

    def frame(self, with_roc_auc: bool = False) -> DataFrame:
        """Get the data used to create the ROC curve plot.

        Parameters
        ----------
        with_roc_auc : bool, default=False
            Whether to include ROC AUC scores in the output DataFrame.

        Returns
        -------
        DataFrame
            A DataFrame containing the ROC curve data with columns depending on the
            report type:

            - `estimator_name`: Name of the estimator (when comparing estimators)
            - `split`: Cross-validation split ID (when doing cross-validation)
            - `label`: Class label (for multiclass-classification)
            - `threshold`: Decision threshold
            - `fpr`: False Positive Rate
            - `tpr`: True Positive Rate
            - `roc_auc`: Area Under the Curve (when `with_roc_auc=True`)

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import EstimatorReport, train_test_split
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> clf = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(clf, **split_data)
        >>> display = report.metrics.roc()
        >>> df = display.frame()
        """
        if with_roc_auc:  # noqa: SIM108
            # The merge between the ROC curve and the ROC AUC is done without
            # specifying the columns to merge on, hence done on all columns that are
            # present in both DataFrames.
            # In this case, the common columns are all columns excepts the ones
            # containing the statistics.
            df = self.roc_curve.merge(self.roc_auc)
        else:
            df = self.roc_curve

        statistical_columns = ["threshold", "fpr", "tpr"]
        if with_roc_auc:
            statistical_columns.append("roc_auc")

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
