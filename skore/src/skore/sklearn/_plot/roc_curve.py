from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer

from skore.sklearn._plot.style import StyleDisplayMixin
from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    _ClassifierCurveDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
    sample_mpl_colormap,
)
from skore.sklearn.types import MLTask


class RocCurveDisplay(
    HelpDisplayMixin, _ClassifierCurveDisplayMixin, StyleDisplayMixin
):
    """ROC Curve visualization.

    An instance of this class is should created by `EstimatorReport.metrics.roc()`.
    You should not create an instance of this class directly.

    Parameters
    ----------
    fpr : dict of list of ndarray
        False positive rate. The structure is:

        - for binary classification:
            - the key is the positive label.
            - the value is a list of `ndarray`, each `ndarray` being the false
              positive rate.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `ndarray`, each `ndarray` being the false
              positive rate.

    tpr : dict of list of ndarray
        True positive rate. The structure is:

        - for binary classification:
            - the key is the positive label
            - the value is a list of `ndarray`, each `ndarray` being the true
              positive rate.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `ndarray`, each `ndarray` being the true
              positive rate.

    roc_auc : dict of list of float
        Area under the ROC curve. The structure is:

        - for binary classification:
            - the key is the positive label
            - the value is a list of `float`, each `float` being the area under
              the ROC curve.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `float`, each `float` being the area under
              the ROC curve.

    estimator_name : str
        Name of the estimator.

    pos_label : int, float, bool, str or None
        The class considered as positive. Only meaningful for binary classification.

    data_source : {"train", "test", "X_y"}
        The data source used to compute the ROC curve.

    Attributes
    ----------
    ax_ : matplotlib axes
        The axes on which the ROC curve is plotted.

    figure_ : matplotlib figure
        The figure on which the ROC curve is plotted.

    lines_ : list of matplotlib lines
        The lines of the ROC curve.

    chance_level_ : matplotlib line
        The chance level line.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from skore import EstimatorReport
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     *load_breast_cancer(return_X_y=True), random_state=0
    ... )
    >>> classifier = LogisticRegression(max_iter=10_000)
    >>> report = EstimatorReport(
    ...     classifier,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ... )
    >>> display = report.metrics.roc()
    >>> display.plot(roc_curve_kwargs={"color": "tab:red"})
    """

    _default_roc_curve_kwargs: Union[dict[str, Any], None] = None
    _default_chance_level_kwargs: Union[dict[str, Any], None] = None

    def __init__(
        self,
        *,
        fpr: dict[Union[int, float, bool, str], list[ArrayLike]],
        tpr: dict[Union[int, float, bool, str], list[ArrayLike]],
        roc_auc: dict[Union[int, float, bool, str], list[float]],
        estimator_name: str,
        pos_label: Union[int, float, bool, str, None],
        data_source: Literal["train", "test", "X_y"],
    ) -> None:
        self.estimator_name = estimator_name
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.pos_label = pos_label
        self.data_source = data_source

    def plot(
        self,
        ax: Optional[Axes] = None,
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
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

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
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_breast_cancer(return_X_y=True), random_state=0
        ... )
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> display = report.metrics.roc()
        >>> display.plot(roc_curve_kwargs={"color": "tab:red"})
        """
        self.ax_, self.figure_, estimator_name = self._validate_plot_params(
            ax=ax, estimator_name=estimator_name
        )

        if roc_curve_kwargs is None:
            roc_curve_kwargs = self._default_roc_curve_kwargs
        if chance_level_kwargs is None:
            chance_level_kwargs = self._default_chance_level_kwargs

        self.lines_: list[Line2D] = []
        default_line_kwargs: dict[str, Any] = {}
        if len(self.fpr) == 1:  # binary-classification
            assert self.pos_label is not None, (
                "pos_label should not be None with binary classification."
            )
            if len(self.fpr[self.pos_label]) == 1:  # single-split
                if roc_curve_kwargs is None:
                    roc_curve_kwargs = {}
                elif isinstance(roc_curve_kwargs, list):
                    if len(roc_curve_kwargs) > 1:
                        raise ValueError(
                            "You intend to plot a single ROC curve and provide "
                            "multiple ROC curve keyword arguments. Provide a single "
                            "dictionary or a list with a single dictionary."
                        )
                    roc_curve_kwargs = roc_curve_kwargs[0]

                fpr = self.fpr[self.pos_label][0]
                tpr = self.tpr[self.pos_label][0]
                roc_auc = self.roc_auc[self.pos_label][0]

                if self.data_source in ("train", "test"):
                    default_line_kwargs["label"] = (
                        f"{self.data_source.title()} set (AUC = {roc_auc:0.2f})"
                    )
                else:  # data_source in (None, "X_y")
                    default_line_kwargs["label"] = f"AUC = {roc_auc:0.2f}"

                line_kwargs = _validate_style_kwargs(
                    default_line_kwargs, roc_curve_kwargs
                )

                (line_,) = self.ax_.plot(fpr, tpr, **line_kwargs)
                self.lines_.append(line_)
            else:  # cross-validation
                if roc_curve_kwargs is None:
                    roc_curve_kwargs = [{}] * len(self.fpr[self.pos_label])
                elif isinstance(roc_curve_kwargs, dict):
                    roc_curve_kwargs = [roc_curve_kwargs] * len(
                        self.fpr[self.pos_label]
                    )
                elif isinstance(roc_curve_kwargs, list):
                    if len(roc_curve_kwargs) != len(self.fpr[self.pos_label]):
                        raise ValueError(
                            "You intend to plot multiple ROC curves. We expect "
                            "`roc_curve_kwargs` to be a list of dictionaries with the "
                            "same length as the number of ROC curves. Got "
                            f"{len(roc_curve_kwargs)} instead of "
                            f"{len(self.fpr[self.pos_label])}."
                        )
                else:
                    raise ValueError(
                        "You intend to plot multiple ROC curves. We expect "
                        "`roc_curve_kwargs` to be a list of dictionaries of "
                        f"{len(self.fpr)} elements. Got {roc_curve_kwargs!r} instead."
                    )

                for split_idx in range(len(self.fpr[self.pos_label])):
                    fpr = self.fpr[self.pos_label][split_idx]
                    tpr = self.tpr[self.pos_label][split_idx]
                    roc_auc = self.roc_auc[self.pos_label][split_idx]

                    default_line_kwargs = {
                        "label": (
                            f"{self.data_source.title()} set - fold #{split_idx + 1} "
                            f"(AUC = {roc_auc:0.2f})"
                        )
                    }
                    line_kwargs = _validate_style_kwargs(
                        default_line_kwargs, roc_curve_kwargs[split_idx]
                    )

                    (line_,) = self.ax_.plot(fpr, tpr, **line_kwargs)
                    self.lines_.append(line_)

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"), 10 if len(self.fpr) < 10 else len(self.fpr)
            )
            if roc_curve_kwargs is None:
                roc_curve_kwargs = [{}] * len(self.fpr)
            elif isinstance(roc_curve_kwargs, list):
                if len(roc_curve_kwargs) != len(self.fpr):
                    raise ValueError(
                        "You intend to plot multiple ROC curves. We expect "
                        "`roc_curve_kwargs` to be a list of dictionaries with the "
                        "same length as the number of ROC curves. Got "
                        f"{len(roc_curve_kwargs)} instead of "
                        f"{len(self.fpr)}."
                    )
            else:
                raise ValueError(
                    "You intend to plot multiple ROC curves. We expect "
                    "`roc_curve_kwargs` to be a list of dictionaries of "
                    f"{len(self.fpr)} elements. Got {roc_curve_kwargs!r} instead."
                )

            for class_idx, class_ in enumerate(self.fpr):
                fpr_class = self.fpr[class_]
                tpr_class = self.tpr[class_]
                roc_auc_class = self.roc_auc[class_]
                roc_curve_kwargs_class = roc_curve_kwargs[class_idx]

                if len(fpr_class) == 1:  # single-split
                    fpr = fpr_class[0]
                    tpr = tpr_class[0]
                    roc_auc = roc_auc_class[0]

                    default_line_kwargs = {"color": class_colors[class_idx]}
                    if self.data_source in ("train", "test"):
                        default_line_kwargs["label"] = (
                            f"{str(class_).title()} - {self.data_source} "
                            f"set (AUC = {roc_auc:0.2f})"
                        )
                    else:  # data_source in (None, "X_y")
                        default_line_kwargs["label"] = (
                            f"{str(class_).title()} - AUC = {roc_auc:0.2f}"
                        )

                    line_kwargs = _validate_style_kwargs(
                        default_line_kwargs, roc_curve_kwargs_class
                    )

                    (line_,) = self.ax_.plot(fpr, tpr, **line_kwargs)
                    self.lines_.append(line_)
                else:  # cross-validation
                    for split_idx in range(len(fpr_class)):
                        fpr = fpr_class[split_idx]
                        tpr = tpr_class[split_idx]
                        roc_auc_mean = np.mean(roc_auc_class)
                        roc_auc_std = np.std(roc_auc_class)

                        default_line_kwargs = {
                            "color": class_colors[class_idx],
                            "alpha": 0.3,
                        }
                        if split_idx == 0:
                            default_line_kwargs["label"] = (
                                f"{str(class_).title()} - {self.data_source} set"
                                f" (AUC = {roc_auc_mean:0.2f} +/- "
                                f"{roc_auc_std:0.2f})"
                            )
                        else:
                            default_line_kwargs["label"] = None

                        line_kwargs = _validate_style_kwargs(default_line_kwargs, {})

                        (line_,) = self.ax_.plot(fpr, tpr, **line_kwargs)
                        self.lines_.append(line_)

        default_chance_level_line_kw = {
            "label": "Chance level (AUC = 0.5)",
            "color": "k",
            "linestyle": "--",
        }

        if chance_level_kwargs is None:
            chance_level_kwargs = {}

        chance_level_kwargs = _validate_style_kwargs(
            default_chance_level_line_kw, chance_level_kwargs
        )

        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        if info_pos_label:
            xlabel += info_pos_label
            ylabel += info_pos_label

        self.ax_.set(
            xlabel=xlabel,
            xlim=(-0.01, 1.01),
            ylabel=ylabel,
            ylim=(-0.01, 1.01),
            aspect="equal",
        )

        self.chance_level_: Optional[Line2D] = None
        if plot_chance_level:
            (self.chance_level_,) = self.ax_.plot((0, 1), (0, 1), **chance_level_kwargs)

        if despine:
            _despine_matplotlib_axis(self.ax_)

        self.ax_.legend(loc="lower right", title=estimator_name)

    @classmethod
    def _from_predictions(
        cls,
        y_true: Sequence[ArrayLike],
        y_pred: Sequence[NDArray],
        *,
        estimator: BaseEstimator,
        estimator_name: str,
        ml_task: MLTask,
        data_source: Literal["train", "test", "X_y"],
        pos_label: Union[int, float, bool, str, None],
        drop_intermediate: bool = True,
    ) -> "RocCurveDisplay":
        """Private method to create a RocCurveDisplay from predictions.

        Parameters
        ----------
        y_true : list of array-like of shape (n_samples,)
            True binary labels in binary classification.

        y_pred : list of array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions (as returned by
            "decision_function" on some classifiers).

        estimator : estimator instance
            The estimator from which `y_pred` is obtained.

        estimator_name : str
            Name of the estimator used to plot the ROC curve.

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

        fpr: dict[Union[int, float, bool, str], list[ArrayLike]] = defaultdict(list)
        tpr: dict[Union[int, float, bool, str], list[ArrayLike]] = defaultdict(list)
        roc_auc: dict[Union[int, float, bool, str], list[float]] = defaultdict(list)

        if ml_task == "binary-classification":
            for y_true_i, y_pred_i in zip(y_true, y_pred):
                fpr_i, tpr_i, _ = roc_curve(
                    y_true_i,
                    y_pred_i,
                    pos_label=pos_label,
                    drop_intermediate=drop_intermediate,
                )
                roc_auc_i = auc(fpr_i, tpr_i)
                # assert for mypy that pos_label_validated is not None
                assert pos_label_validated is not None, (
                    "pos_label_validated should not be None with binary classification "
                    "once calling _validate_from_predictions_params and more precisely "
                    "_check_pos_label_consistency."
                )
                fpr[pos_label_validated].append(fpr_i)
                tpr[pos_label_validated].append(tpr_i)
                roc_auc[pos_label_validated].append(roc_auc_i)
        else:  # multiclass-classification
            # OvR fashion to collect fpr, tpr, and roc_auc
            for y_true_i, y_pred_i in zip(y_true, y_pred):
                label_binarizer = LabelBinarizer().fit(estimator.classes_)
                y_true_onehot_i: NDArray = label_binarizer.transform(y_true_i)
                for class_idx, class_ in enumerate(estimator.classes_):
                    fpr_class_i, tpr_class_i, _ = roc_curve(
                        y_true_onehot_i[:, class_idx],
                        y_pred_i[:, class_idx],
                        pos_label=None,
                        drop_intermediate=drop_intermediate,
                    )
                    roc_auc_class_i = auc(fpr_class_i, tpr_class_i)

                    fpr[class_].append(fpr_class_i)
                    tpr[class_].append(tpr_class_i)
                    roc_auc[class_].append(roc_auc_class_i)

        return cls(
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc,
            estimator_name=estimator_name,
            pos_label=pos_label_validated,
            data_source=data_source,
        )
