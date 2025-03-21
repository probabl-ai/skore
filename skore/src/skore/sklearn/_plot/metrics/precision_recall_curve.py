from collections import defaultdict
from typing import Any, Literal, Optional, Union

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, precision_recall_curve
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


class PrecisionRecallCurveDisplay(
    HelpDisplayMixin, _ClassifierCurveDisplayMixin, StyleDisplayMixin
):
    """Precision Recall visualization.

    An instance of this class is should created by
    `EstimatorReport.metrics.precision_recall()`. You should not create an
    instance of this class directly.


    Parameters
    ----------
    precision : dict of list of ndarray
        Precision values. The structure is:

        - for binary classification:
            - the key is the positive label.
            - the value is a list of `ndarray`, each `ndarray` being the precision.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `ndarray`, each `ndarray` being the precision.

    recall : dict of list of ndarray
        Recall values. The structure is:

        - for binary classification:
            - the key is the positive label.
            - the value is a list of `ndarray`, each `ndarray` being the recall.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `ndarray`, each `ndarray` being the recall.

    average_precision : dict of list of float
        Average precision. The structure is:

        - for binary classification:
            - the key is the positive label.
            - the value is a list of `float`, each `float` being the average
              precision.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `float`, each `float` being the average
              precision.

    estimator_name : str
        Name of the estimator.

    pos_label : int, float, bool, str or None
        The class considered as the positive class. If None, the class will not
        be shown in the legend.

    data_source : {"train", "test", "X_y"}
        The data source used to compute the precision recall curve.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with precision recall curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    lines_ : list of matplotlib Artist
        Precision recall curve.

    chance_levels_ : matplotlib Artist or None
        The chance level line. It is `None` if the chance level is not plotted.

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
    >>> display = report.metrics.precision_recall()
    >>> display.plot(pr_curve_kwargs={"color": "tab:red"})
    """

    _default_pr_curve_kwargs: Union[dict[str, Any], None] = None

    def __init__(
        self,
        *,
        precision: dict[Any, list[ArrayLike]],
        recall: dict[Any, list[ArrayLike]],
        average_precision: dict[Any, list[float]],
        estimator_name: str,
        pos_label: Union[int, float, bool, str, None],
        data_source: Literal["train", "test", "X_y"],
    ) -> None:
        self.precision = precision
        self.recall = recall
        self.average_precision = average_precision
        self.estimator_name = estimator_name
        self.pos_label = pos_label
        self.data_source = data_source

    def plot(
        self,
        ax: Optional[Axes] = None,
        *,
        estimator_name: Optional[str] = None,
        pr_curve_kwargs: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
        despine: bool = True,
    ) -> None:
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        ax : Matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

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
        >>> display = report.metrics.precision_recall()
        >>> display.plot(pr_curve_kwargs={"color": "tab:red"})
        """
        self.ax_, self.figure_, estimator_name = self._validate_plot_params(
            ax=ax, estimator_name=estimator_name
        )

        if pr_curve_kwargs is None:
            pr_curve_kwargs = self._default_pr_curve_kwargs

        self.lines_ = []
        default_line_kwargs: dict[str, Any] = {}
        if len(self.precision) == 1:  # binary-classification
            assert self.pos_label is not None, (
                "pos_label should not be None with binary classification."
            )
            if len(self.precision[self.pos_label]) == 1:  # single-split
                if pr_curve_kwargs is None:
                    pr_curve_kwargs = {}
                elif isinstance(pr_curve_kwargs, list):
                    if len(pr_curve_kwargs) > 1:
                        raise ValueError(
                            "You intend to plot a single precision-recall curve and "
                            "provide multiple precision-recall curve keyword "
                            "arguments. Provide a single dictionary or a list with "
                            "a single dictionary."
                        )
                    pr_curve_kwargs = pr_curve_kwargs[0]

                precision = self.precision[self.pos_label][0]
                recall = self.recall[self.pos_label][0]
                average_precision = self.average_precision[self.pos_label][0]

                default_line_kwargs = {"drawstyle": "steps-post"}
                if self.data_source in ("train", "test"):
                    default_line_kwargs["label"] = (
                        f"{self.data_source.title()} set "
                        f"(AP = {average_precision:0.2f})"
                    )
                else:  # data_source in (None, "X_y")
                    default_line_kwargs["label"] = f"AP = {average_precision:0.2f}"

                line_kwargs = _validate_style_kwargs(
                    default_line_kwargs, pr_curve_kwargs
                )

                (line_,) = self.ax_.plot(recall, precision, **line_kwargs)
                self.lines_.append(line_)
            else:  # cross-validation
                if pr_curve_kwargs is None:
                    pr_curve_kwargs = [{}] * len(self.precision[self.pos_label])
                elif isinstance(pr_curve_kwargs, list):
                    if len(pr_curve_kwargs) != len(self.precision[self.pos_label]):
                        raise ValueError(
                            "You intend to plot multiple precision-recall curves. We "
                            "expect `pr_curve_kwargs` to be a list of dictionaries "
                            "with the same length as the number of precision-recall "
                            "curves. Got "
                            f"{len(pr_curve_kwargs)} instead of "
                            f"{len(self.precision)}."
                        )
                else:
                    raise ValueError(
                        "You intend to plot multiple precision-recall curves. We "
                        "expect `pr_curve_kwargs` to be a list of dictionaries of "
                        f"{len(self.precision)} elements. Got {pr_curve_kwargs!r} "
                        "instead."
                    )

                for split_idx in range(len(self.precision[self.pos_label])):
                    precision = self.precision[self.pos_label][split_idx]
                    recall = self.recall[self.pos_label][split_idx]
                    average_precision = self.average_precision[self.pos_label][
                        split_idx
                    ]

                    default_line_kwargs = {
                        "drawstyle": "steps-post",
                        "label": (
                            f"{self.data_source.title()} set - fold #{split_idx + 1} "
                            f"(AP = {average_precision:0.2f})"
                        ),
                    }
                    line_kwargs = _validate_style_kwargs(
                        default_line_kwargs, pr_curve_kwargs[split_idx]
                    )

                    (line_,) = self.ax_.plot(recall, precision, **line_kwargs)
                    self.lines_.append(line_)

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
            class_colors = sample_mpl_colormap(
                colormaps.get_cmap("tab10"),
                10 if len(self.precision) < 10 else len(self.precision),
            )
            if pr_curve_kwargs is None:
                pr_curve_kwargs = [{}] * len(self.precision)
            elif isinstance(pr_curve_kwargs, list):
                if len(pr_curve_kwargs) != len(self.precision):
                    raise ValueError(
                        "You intend to plot multiple precision-recall curves. We "
                        "expect `pr_curve_kwargs` to be a list of dictionaries with "
                        "the same length as the number of precision-recall curves. "
                        "Got "
                        f"{len(pr_curve_kwargs)} instead of "
                        f"{len(self.precision)}."
                    )
            else:
                raise ValueError(
                    "You intend to plot multiple precision-recall curves. We expect "
                    "`pr_curve_kwargs` to be a list of dictionaries of "
                    f"{len(self.precision)} elements. Got {pr_curve_kwargs!r} instead."
                )

            for class_idx, class_ in enumerate(self.precision):
                precision_class = self.precision[class_]
                recall_class = self.recall[class_]
                average_precision_class = self.average_precision[class_]
                pr_curve_kwargs_class = pr_curve_kwargs[class_idx]

                if len(precision_class) == 1:  # single-split
                    precision = precision_class[0]
                    recall = recall_class[0]
                    average_precision = average_precision_class[0]

                    default_line_kwargs = {
                        "drawstyle": "steps-post",
                        "color": class_colors[class_idx],
                    }
                    if self.data_source in ("train", "test"):
                        default_line_kwargs["label"] = (
                            f"{str(class_).title()} - {self.data_source} set "
                            f"(AP = {average_precision:0.2f})"
                        )
                    else:  # data_source in (None, "X_y")
                        default_line_kwargs["label"] = (
                            f"{str(class_).title()} AP = {average_precision:0.2f}"
                        )

                    line_kwargs = _validate_style_kwargs(
                        default_line_kwargs, pr_curve_kwargs_class
                    )

                    (line_,) = self.ax_.plot(recall, precision, **line_kwargs)
                    self.lines_.append(line_)
                else:  # cross-validation
                    for split_idx in range(len(precision_class)):
                        precision = precision_class[split_idx]
                        recall = recall_class[split_idx]
                        average_precision = average_precision_class[split_idx]

                        default_line_kwargs = {
                            "color": class_colors[class_idx],
                            "alpha": 0.3,
                        }
                        if split_idx == 0:
                            default_line_kwargs["label"] = (
                                f"{str(class_).title()} - {self.data_source} set"
                                f" (AP = {np.mean(average_precision_class):0.2f} +/- "
                                f"{np.std(average_precision_class):0.2f})"
                            )
                        else:
                            default_line_kwargs["label"] = None

                        line_kwargs = _validate_style_kwargs(default_line_kwargs, {})

                        (line_,) = self.ax_.plot(recall, precision, **line_kwargs)
                        self.lines_.append(line_)

        xlabel = "Recall"
        ylabel = "Precision"
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

        if despine:
            _despine_matplotlib_axis(self.ax_)

        self.ax_.legend(loc="lower left", title=estimator_name)

    @classmethod
    def _from_predictions(
        cls,
        y_true: list[ArrayLike],
        y_pred: list[NDArray],
        *,
        estimator: BaseEstimator,
        estimator_name: str,
        ml_task: MLTask,
        data_source: Literal["train", "test", "X_y"],
        pos_label: Union[int, float, bool, str, None],
        drop_intermediate: bool = False,
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

        estimator : estimator instance
            The estimator from which `y_pred` is obtained.

        estimator_name : str
            Name of the estimator used to plot the precision-recall curve.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y"}
            The data source used to compute the precision recall curve.

        pos_label : int, float, bool, str or none
            The class considered as the positive class when computing the
            precision and recall metrics.

        drop_intermediate : bool, default=False
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

        precision: dict[Union[int, float, bool, str], list[ArrayLike]] = defaultdict(
            list
        )
        recall: dict[Union[int, float, bool, str], list[ArrayLike]] = defaultdict(list)
        average_precision: dict[Union[int, float, bool, str], list[float]] = (
            defaultdict(list)
        )

        if ml_task == "binary-classification":
            for y_true_i, y_pred_i in zip(y_true, y_pred):
                # assert for mypy that pos_label_validated is not None
                assert pos_label_validated is not None, (
                    "pos_label_validated should not be None with binary classification "
                    "once calling _validate_from_predictions_params and more precisely "
                    "_check_pos_label_consistency."
                )
                precision_i, recall_i, _ = precision_recall_curve(
                    y_true_i,
                    y_pred_i,
                    pos_label=pos_label_validated,
                    drop_intermediate=drop_intermediate,
                )
                average_precision_i = average_precision_score(
                    y_true_i, y_pred_i, pos_label=pos_label_validated
                )

                precision[pos_label_validated].append(precision_i)
                recall[pos_label_validated].append(recall_i)
                average_precision[pos_label_validated].append(average_precision_i)
        else:  # multiclass-classification
            for y_true_i, y_pred_i in zip(y_true, y_pred):
                label_binarizer = LabelBinarizer().fit(estimator.classes_)
                y_true_onehot_i: NDArray = label_binarizer.transform(y_true_i)
                for class_idx, class_ in enumerate(estimator.classes_):
                    precision_class_i, recall_class_i, _ = precision_recall_curve(
                        y_true_onehot_i[:, class_idx],
                        y_pred_i[:, class_idx],
                        pos_label=None,
                        drop_intermediate=drop_intermediate,
                    )
                    average_precision_class_i = average_precision_score(
                        y_true_onehot_i[:, class_idx], y_pred_i[:, class_idx]
                    )

                    precision[class_].append(precision_class_i)
                    recall[class_].append(recall_class_i)
                    average_precision[class_].append(average_precision_class_i)

        return cls(
            precision=precision,
            recall=recall,
            average_precision=average_precision,
            estimator_name=estimator_name,
            pos_label=pos_label_validated,
            data_source=data_source,
        )
