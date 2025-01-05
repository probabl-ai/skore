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

    estimator_name : str, default=None
        Name of the estimator.

    pos_label : str, default=None
        The class considered as positive. Only meaningful for binary classification.

    data_source : {"train", "test", "X_y"}, default=None
        The data source used to compute the ROC curve.
    """

    def __init__(
        self,
        *,
        fpr,
        tpr,
        roc_auc=None,
        estimator_name=None,
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
        name=None,
        plot_chance_level=False,
        chance_level_kw=None,
        despine=False,
        **kwargs,
    ):
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of ROC Curve for labeling. If `None`, use `estimator_name` if
            not `None`, otherwise no labeling is shown.

        plot_chance_level : bool, default=False
            Whether to plot the chance level.

        chance_level_kw : dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the chance level line.

        despine : bool, default=False
            Whether to remove the top and right spines from the plot.

        **kwargs : dict
            Keyword arguments to be passed to matplotlib's `plot`.

        Returns
        -------
        display : :class:`~sklearn.metrics.RocCurveDisplay`
            Object that stores computed values.
        """
        self.ax_, self.figure_, name = self._validate_plot_params(ax=ax, name=name)

        self.lines_ = []
        if len(self.fpr) == 1:  # binary-classification
            if len(self.fpr[self.pos_label]) == 1:  # single-split
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

                line_kwargs = _validate_style_kwargs(default_line_kwargs, kwargs)

                (line_,) = self.ax_.plot(fpr, tpr, **line_kwargs)
                self.lines_.append(line_)

            info_pos_label = (
                f"\n(Positive label: {self.pos_label})"
                if self.pos_label is not None
                else ""
            )
        else:  # multiclass-classification
            info_pos_label = None  # irrelevant for multiclass

            for class_ in self.fpr:
                fpr_class = self.fpr[class_]
                tpr_class = self.tpr[class_]
                roc_auc_class = self.roc_auc[class_]

                if len(fpr_class) == 1:  # single-split
                    fpr = fpr_class[0]
                    tpr = tpr_class[0]
                    roc_auc = roc_auc_class[0]

                    default_line_kwargs = {}
                    if roc_auc is not None and self.data_source in ("train", "test"):
                        default_line_kwargs["label"] = (
                            f"{str(class_).title()} {self.data_source} "
                            f"set (AUC = {roc_auc:0.2f})"
                        )
                    elif roc_auc is not None:  # data_source in (None, "X_y")
                        default_line_kwargs["label"] = (
                            f"{str(class_).title()} AUC = {roc_auc:0.2f}"
                        )

                    line_kwargs = _validate_style_kwargs(default_line_kwargs, kwargs)

                    (line_,) = self.ax_.plot(fpr, tpr, **line_kwargs)
                    self.lines_.append(line_)

        default_chance_level_line_kw = {
            "label": "Chance level (AUC = 0.5)",
            "color": "k",
            "linestyle": "--",
        }

        if chance_level_kw is None:
            chance_level_kw = {}

        chance_level_kw = _validate_style_kwargs(
            default_chance_level_line_kw, chance_level_kw
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
            (self.chance_level_,) = self.ax_.plot((0, 1), (0, 1), **chance_level_kw)
        else:
            self.chance_level_ = None

        if despine:
            _despine_matplotlib_axis(self.ax_)

        if "label" in line_kwargs or "label" in chance_level_kw:
            self.ax_.legend(loc="lower right", title=name)

        return self

    @classmethod
    def _from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        estimator,
        ml_task,
        data_source=None,
        sample_weight=None,
        drop_intermediate=True,
        pos_label=None,
        name=None,
        ax=None,
        plot_chance_level=False,
        chance_level_kw=None,
        despine=False,
        **kwargs,
    ):
        pos_label_validated, name = cls._validate_from_predictions_params(
            y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label, name=name
        )

        if ml_task == "binary-classification":
            fpr, tpr, _ = roc_curve(
                y_true,
                y_pred,
                pos_label=pos_label,
                sample_weight=sample_weight,
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
                    sample_weight=sample_weight,
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
            estimator_name=name,
            pos_label=pos_label_validated,
            data_source=data_source,
        )

        return viz.plot(
            ax=ax,
            name=name,
            plot_chance_level=plot_chance_level,
            chance_level_kw=chance_level_kw,
            despine=despine,
            **kwargs,
        )
