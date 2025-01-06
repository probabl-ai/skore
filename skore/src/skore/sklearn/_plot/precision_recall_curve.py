from collections import Counter

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer

from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    _ClassifierCurveDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
)


class PrecisionRecallDisplay(HelpDisplayMixin, _ClassifierCurveDisplayMixin):
    """Precision Recall visualization.

    An instance of this class is should created by
    `EstimatorReport.metrics.plot.precision_recall()`. You should not create an
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

    prevalence : dict of list of float
        The prevalence of the positive label. The structure is:

        - for binary classification:
            - the key is the positive label.
            - the value is a list of `float`, each `float` being the prevalence.
        - for multiclass classification:
            - the key is the class of interest in an OvR fashion.
            - the value is a list of `float`, each `float` being the prevalence.

    estimator_name : str, default=None
        Name of estimator. If None, then the estimator name is not shown.

    pos_label : int, float, bool or str, default=None
        The class considered as the positive class. If None, the class will not
        be shown in the legend.

    data_source : {"train", "test", "X_y"}, default=None
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

    """

    def __init__(
        self,
        precision,
        recall,
        *,
        average_precision=None,
        prevalence=None,
        estimator_name=None,
        pos_label=None,
        data_source=None,
    ):
        self.precision = precision
        self.recall = recall
        self.average_precision = average_precision
        self.prevalence = prevalence
        self.estimator_name = estimator_name
        self.pos_label = pos_label
        self.data_source = data_source

    def plot(
        self,
        ax=None,
        *,
        name=None,
        pr_curve_kwargs=None,
        plot_chance_level=False,
        chance_level_kwargs=None,
        despine=True,
    ):
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        ax : Matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of precision recall curve for labeling. If `None`, use
            `estimator_name` if not `None`, otherwise no labeling is shown.

        plot_chance_level : bool, default=True
            Whether to plot the chance level. The chance level is the prevalence
            of the positive label computed from the data passed during
            :meth:`from_estimator` or :meth:`from_predictions` call.

        pr_curve_kwargs : dict or list of dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the precision-recall curve(s).

        chance_level_kwargs : dict or list of dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the chance level line.

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

        Returns
        -------
        display : :class:`~sklearn.metrics.PrecisionRecallDisplay`
            Object that stores computed values.

        Notes
        -----
        The average precision (cf. :func:`~sklearn.metrics.average_precision_score`)
        in scikit-learn is computed without any interpolation. To be consistent
        with this metric, the precision-recall curve is plotted without any
        interpolation as well (step-wise style).

        You can change this style by passing the keyword argument
        `drawstyle="default"`. However, the curve will not be strictly
        consistent with the reported average precision.
        """
        self.ax_, self.figure_, name = self._validate_plot_params(ax=ax, name=name)

        self.lines_ = []
        self.chance_levels_ = []
        if len(self.precision) == 1:  # binary-classification
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
                prevalence = self.prevalence[self.pos_label][0]

                default_line_kwargs = {"drawstyle": "steps-post"}
                if average_precision is not None and self.data_source in (
                    "train",
                    "test",
                ):
                    default_line_kwargs["label"] = (
                        f"{self.data_source.title()} set "
                        f"(AP = {average_precision:0.2f})"
                    )
                elif average_precision is not None:  # data_source in (None, "X_y")
                    default_line_kwargs["label"] = f"AP = {average_precision:0.2f}"

                line_kwargs = _validate_style_kwargs(
                    default_line_kwargs, pr_curve_kwargs
                )

                (line_,) = self.ax_.plot(recall, precision, **line_kwargs)
                self.lines_.append(line_)

                if plot_chance_level:
                    default_chance_level_line_kwargs = {
                        "label": f"Chance level (AP = {prevalence:0.2f})",
                        "color": "k",
                        "linestyle": "--",
                    }

                    if chance_level_kwargs is None:
                        chance_level_kwargs = {}
                    elif isinstance(chance_level_kwargs, list):
                        if len(chance_level_kwargs) > 1:
                            raise ValueError(
                                "You intend to plot a single chance level line and "
                                "provide multiple chance level line keyword "
                                "arguments. Provide a single dictionary or a list "
                                "with a single dictionary."
                            )
                        chance_level_kwargs = chance_level_kwargs[0]

                    chance_level_line_kwargs = _validate_style_kwargs(
                        default_chance_level_line_kwargs, chance_level_kwargs
                    )

                    (chance_level_,) = self.ax_.plot(
                        (0, 1), (prevalence, prevalence), **chance_level_line_kwargs
                    )
                    self.chance_levels_.append(chance_level_)
                else:
                    self.chance_levels_ = None

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
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

            if plot_chance_level:
                if chance_level_kwargs is None:
                    chance_level_kwargs = [{}] * len(self.precision)
                elif isinstance(chance_level_kwargs, list):
                    if len(chance_level_kwargs) != len(self.precision):
                        raise ValueError(
                            "You intend to plot multiple precision-recall curves. We "
                            "expect `chance_level_kwargs` to be a list of dictionaries "
                            "with the same length as the number of precision-recall "
                            "curves. Got "
                            f"{len(chance_level_kwargs)} instead of "
                            f"{len(self.precision)}."
                        )
                else:
                    raise ValueError(
                        "You intend to plot multiple precision-recall curves. We "
                        "expect `chance_level_kwargs` to be a list of dictionaries of "
                        f"{len(self.precision)} elements. Got {chance_level_kwargs!r} "
                        "instead."
                    )

            for class_idx, class_ in enumerate(self.precision):
                precision_class = self.precision[class_]
                recall_class = self.recall[class_]
                average_precision_class = self.average_precision[class_]
                prevalence_class = self.prevalence[class_]
                pr_curve_kwargs_class = pr_curve_kwargs[class_idx]

                if len(precision_class) == 1:  # single-split
                    precision = precision_class[0]
                    recall = recall_class[0]
                    average_precision = average_precision_class[0]
                    prevalence = prevalence_class[0]

                    default_line_kwargs = {"drawstyle": "steps-post"}
                    if average_precision is not None and self.data_source in (
                        "train",
                        "test",
                    ):
                        default_line_kwargs["label"] = (
                            f"{str(class_).title()} - {self.data_source} set "
                            f"(AP = {average_precision:0.2f})"
                        )
                    elif average_precision is not None:  # data_source in (None, "X_y")
                        default_line_kwargs["label"] = (
                            f"{str(class_).title()} AP = {average_precision:0.2f}"
                        )

                    line_kwargs = _validate_style_kwargs(
                        default_line_kwargs, pr_curve_kwargs_class
                    )

                    (line_,) = self.ax_.plot(recall, precision, **line_kwargs)
                    self.lines_.append(line_)

                    if plot_chance_level:
                        chance_level_kwargs_class = chance_level_kwargs[class_idx]

                        default_chance_level_line_kwargs = {
                            "label": (
                                f"Chance level - {str(class_).title()} "
                                f"(AP = {prevalence:0.2f})"
                            ),
                            "color": "k",
                            "linestyle": "--",
                        }

                        chance_level_line_kwargs = _validate_style_kwargs(
                            default_chance_level_line_kwargs, chance_level_kwargs_class
                        )

                        (chance_level_,) = self.ax_.plot(
                            (0, 1), (prevalence, prevalence), **chance_level_line_kwargs
                        )
                        self.chance_levels_.append(chance_level_)
                    else:
                        self.chance_levels_ = None

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

        if "label" in line_kwargs or plot_chance_level:
            self.ax_.legend(loc="lower left", title=name)

    @classmethod
    def _from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        estimator,
        ml_task,
        data_source=None,
        pos_label=None,
        drop_intermediate=False,
        ax=None,
        name=None,
        pr_curve_kwargs=None,
        plot_chance_level=False,
        chance_level_kwargs=None,
        despine=True,
    ):
        """Plot precision-recall curve given binary class predictions.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels.

        y_pred : array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions (as returned by
            “decision_function” on some classifiers).

        estimator : estimator instance
            The estimator from which `y_pred` is obtained.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y"}, default=None
            The data source used to compute the ROC curve.

        pos_label : int, float, bool or str, default=None
            The class considered as the positive class when computing the
            precision and recall metrics.

        drop_intermediate : bool, default=False
            Whether to drop some suboptimal thresholds which would not appear
            on a plotted precision-recall curve. This is useful in order to
            create lighter precision-recall curves.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is created.

        name : str, default=None
            Name for labeling curve. If `None`, name will be set to
            `"Classifier"`.

        pr_curve_kwargs : dict or list of dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the precision-recall curve(s).

        plot_chance_level : bool, default=False
            Whether to plot the chance level. The chance level is the prevalence
            of the positive label computed from the data passed during
            :meth:`from_estimator` or :meth:`from_predictions` call.

        chance_level_kwargs : dict or list of dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the chance level line.

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

        **kwargs : dict
            Keyword arguments to be passed to matplotlib's `plot`.

        Returns
        -------
        display : :class:`~sklearn.metrics.PrecisionRecallDisplay`
        """
        pos_label_validated, name = cls._validate_from_predictions_params(
            y_true, y_pred, pos_label=pos_label, name=name
        )

        if ml_task == "binary-classification":
            precision, recall, _ = precision_recall_curve(
                y_true,
                y_pred,
                pos_label=pos_label_validated,
                drop_intermediate=drop_intermediate,
            )
            average_precision = average_precision_score(
                y_true, y_pred, pos_label=pos_label_validated
            )

            class_count = Counter(y_true)
            prevalence = class_count[pos_label_validated] / sum(class_count.values())

            precision = {pos_label_validated: [precision]}
            recall = {pos_label_validated: [recall]}
            average_precision = {pos_label_validated: [average_precision]}
            prevalence = {pos_label_validated: [prevalence]}
        else:  # multiclass-classification
            precision, recall, average_precision, prevalence = {}, {}, {}, {}
            label_binarizer = LabelBinarizer().fit(estimator.classes_)
            y_true_onehot = label_binarizer.transform(y_true)
            for class_idx, class_ in enumerate(estimator.classes_):
                precision_class, recall_class, _ = precision_recall_curve(
                    y_true_onehot[:, class_idx],
                    y_pred[:, class_idx],
                    pos_label=None,
                    drop_intermediate=drop_intermediate,
                )
                average_precision_class = average_precision_score(
                    y_true_onehot[:, class_idx], y_pred[:, class_idx]
                )
                class_count = Counter(y_true)
                prevalence_class = class_count[class_] / sum(class_count.values())

                precision[class_] = [precision_class]
                recall[class_] = [recall_class]
                average_precision[class_] = [average_precision_class]
                prevalence[class_] = [prevalence_class]

        viz = cls(
            precision=precision,
            recall=recall,
            average_precision=average_precision,
            prevalence=prevalence,
            estimator_name=name,
            pos_label=pos_label_validated,
            data_source=data_source,
        )

        viz.plot(
            ax=ax,
            name=name,
            pr_curve_kwargs=pr_curve_kwargs,
            plot_chance_level=plot_chance_level,
            chance_level_kwargs=chance_level_kwargs,
            despine=despine,
        )

        return viz
