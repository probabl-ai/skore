from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer

from skore.sklearn._plot.utils import (
    HelpDisplayMixin,
    _ClassifierCurveDisplayMixin,
    _despine_matplotlib_axis,
    _validate_style_kwargs,
)


class RocCurveDisplay(HelpDisplayMixin, _ClassifierCurveDisplayMixin):
    """ROC Curve visualization.

    An instance of this class is should created by `EstimatorReport.metrics.plot.roc()`.
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

    pos_label : str, default=None
        The class considered as positive. Only meaningful for binary classification.

    data_source : {"train", "test", "X_y"}, default=None
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
    """

    def __init__(
        self,
        *,
        fpr,
        tpr,
        roc_auc,
        estimator_name,
        pos_label=None,
        data_source=None,
    ):
        self.estimator_name = estimator_name
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.pos_label = pos_label
        self.data_source = data_source

    def plot(
        self,
        ax=None,
        *,
        estimator_name=None,
        roc_curve_kwargs=None,
        plot_chance_level=True,
        chance_level_kwargs=None,
        despine=True,
    ):
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

        Returns
        -------
        display : :class:`~sklearn.metrics.RocCurveDisplay`
            Object that stores computed values.
        """
        self.ax_, self.figure_, estimator_name = self._validate_plot_params(
            ax=ax, estimator_name=estimator_name
        )

        self.lines_ = []
        if len(self.fpr) == 1:  # binary-classification
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

                default_line_kwargs = {}
                if roc_auc is not None and self.data_source in ("train", "test"):
                    default_line_kwargs["label"] = (
                        f"{self.data_source.title()} set (AUC = {roc_auc:0.2f})"
                    )
                elif roc_auc is not None:  # data_source in (None, "X_y")
                    default_line_kwargs["label"] = f"AUC = {roc_auc:0.2f}"

                line_kwargs = _validate_style_kwargs(
                    default_line_kwargs, roc_curve_kwargs
                )

                (line_,) = self.ax_.plot(fpr, tpr, **line_kwargs)
                self.lines_.append(line_)
            else:  # cross-validation
                raise NotImplementedError(
                    "We don't support yet cross-validation"
                )  # pragma: no cover

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass
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

                    default_line_kwargs = {}
                    if roc_auc is not None and self.data_source in ("train", "test"):
                        default_line_kwargs["label"] = (
                            f"{str(class_).title()} - {self.data_source} "
                            f"set (AUC = {roc_auc:0.2f})"
                        )
                    elif roc_auc is not None:  # data_source in (None, "X_y")
                        default_line_kwargs["label"] = (
                            f"{str(class_).title()} AUC = {roc_auc:0.2f}"
                        )

                    line_kwargs = _validate_style_kwargs(
                        default_line_kwargs, roc_curve_kwargs_class
                    )

                    (line_,) = self.ax_.plot(fpr, tpr, **line_kwargs)
                    self.lines_.append(line_)
                else:  # cross-validation
                    raise NotImplementedError(
                        "We don't support yet cross-validation"
                    )  # pragma: no cover

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

        if plot_chance_level:
            (self.chance_level_,) = self.ax_.plot((0, 1), (0, 1), **chance_level_kwargs)
        else:
            self.chance_level_ = None

        if despine:
            _despine_matplotlib_axis(self.ax_)

        self.ax_.legend(loc="lower right", title=estimator_name)

    @classmethod
    def _from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        estimator,
        estimator_name,
        ml_task,
        data_source=None,
        pos_label=None,
        drop_intermediate=True,
        ax=None,
        roc_curve_kwargs=None,
        plot_chance_level=True,
        chance_level_kwargs=None,
        despine=True,
    ):
        """Private method to create a RocCurveDisplay from predictions.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels in binary classification.

        y_pred : array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions (as returned by
            “decision_function” on some classifiers).

        estimator : estimator instance
            The estimator from which `y_pred` is obtained.

        estimator_name : str
            Name of the estimator used to plot the ROC curve.

        ml_task : {"binary-classification", "multiclass-classification"}
            The machine learning task.

        data_source : {"train", "test", "X_y"}, default=None
            The data source used to compute the ROC curve.

        pos_label : int, float, bool or str, default=None
            The class considered as the positive class when computing the
            precision and recall metrics.

        drop_intermediate : bool, default=True
            Whether to drop intermediate points with identical value.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

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

        Returns
        -------
        display : RocCurveDisplay
            Object that stores computed values.
        """
        pos_label_validated = cls._validate_from_predictions_params(
            y_true, y_pred, ml_task=ml_task, pos_label=pos_label
        )

        if ml_task == "binary-classification":
            fpr, tpr, _ = roc_curve(
                y_true,
                y_pred,
                pos_label=pos_label,
                drop_intermediate=drop_intermediate,
            )
            roc_auc = auc(fpr, tpr)
            fpr = {pos_label_validated: [fpr]}
            tpr = {pos_label_validated: [tpr]}
            roc_auc = {pos_label_validated: [roc_auc]}
        else:  # multiclass-classification
            # OvR fashion to collect fpr, tpr, and roc_auc
            fpr, tpr, roc_auc = {}, {}, {}
            label_binarizer = LabelBinarizer().fit(estimator.classes_)
            y_true_onehot = label_binarizer.transform(y_true)
            for class_idx, class_ in enumerate(estimator.classes_):
                fpr_class, tpr_class, _ = roc_curve(
                    y_true_onehot[:, class_idx],
                    y_pred[:, class_idx],
                    pos_label=None,
                    drop_intermediate=drop_intermediate,
                )
                roc_auc_class = auc(fpr_class, tpr_class)

                fpr[class_] = [fpr_class]
                tpr[class_] = [tpr_class]
                roc_auc[class_] = [roc_auc_class]

        viz = cls(
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc,
            estimator_name=estimator_name,
            pos_label=pos_label_validated,
            data_source=data_source,
        )

        viz.plot(
            ax=ax,
            estimator_name=estimator_name,
            roc_curve_kwargs=roc_curve_kwargs,
            plot_chance_level=plot_chance_level,
            chance_level_kwargs=chance_level_kwargs,
            despine=despine,
        )

        return viz
