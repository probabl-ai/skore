from collections.abc import Sequence
from typing import Any, Literal, cast

import seaborn as sns
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import (
    _build_custom_legend_with_stats,
    _ClassifierDisplayMixin,
    _despine_matplotlib_axis,
    _get_curve_plot_columns,
    _validate_style_kwargs,
)
from skore._sklearn.types import (
    DataSource,
    MLTask,
    PositiveLabel,
    ReportType,
    YPlotData,
)


class RocCurveDisplay(_ClassifierDisplayMixin, DisplayMixin):
    """ROC Curve visualization.

    An instance of this class should be created by `EstimatorReport.metrics.roc()`.
    You should not create an instance of this class directly.

    Parameters
    ----------
    roc_curve : DataFrame
        The ROC curve data to display. The columns are

        - `estimator`
        - `split` (may be null)
        - `label`
        - `threshold`
        - `fpr`
        - `tpr`.

    roc_auc : DataFrame
        The ROC AUC data to display. The columns are

        - `estimator`
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
    facet_ : seaborn FacetGrid
        FacetGrid containing the ROC curve.

    figure_ : matplotlib figure
        The figure on which the ROC curve is plotted.

    ax_ : matplotlib axes or array of axes
        The axes on which the ROC curve is plotted.

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
    >>> display.set_style(relplot_kwargs={"color": "tab:red"}).plot()
    """

    _default_relplot_kwargs = {
        "height": 6,
        "aspect": 1,
        "facet_kws": {
            "sharex": False,
            "sharey": False,
            "xlim": (-0.01, 1.01),
            "ylim": (-0.01, 1.01),
        },
        "drawstyle": "steps-post",
        "legend": False,
    }
    _default_chance_level_kwargs = {
        "color": "k",
        "linestyle": "--",
    }

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

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        subplot_by: Literal["auto", "label", "estimator", "data_source"]
        | None = "auto",
        plot_chance_level: bool = True,
        despine: bool = True,
    ) -> None:
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        subplot_by : {"auto", "label", "estimator", "data_source"} or None, \
            default="auto"
            Column to use for creating subplots. Options:

            - "auto": None for :class:`~skore.EstimatorReport` and \
              :class:`~skore.CrossValidationReport`, "estimator" for \
              :class:`~skore.ComparisonReport`
            - "label": one subplot per class (multiclass only)
            - "estimator": one subplot per estimator (comparison only)
            - "data_source": one subplot per data source \
              (:class:`~skore.EstimatorReport` with both data sources only)
            - None: no subplots (not available for comparison in multiclass)

        plot_chance_level : bool, default=True
            Whether to plot the chance level.

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
        >>> display.set_style(relplot_kwargs={"color": "tab:red"}).plot()
        """
        return self._plot(
            subplot_by=subplot_by,
            plot_chance_level=plot_chance_level,
            despine=despine,
        )

    def _plot_matplotlib(
        self,
        *,
        subplot_by: Literal["auto", "label", "estimator", "data_source"]
        | None = "auto",
        plot_chance_level: bool = True,
        despine: bool = True,
    ) -> None:
        """Matplotlib implementation of the `plot` method."""
        plot_data = self.frame(with_roc_auc=True)

        col, hue, style = _get_curve_plot_columns(
            plot_data=plot_data,
            report_type=self.report_type,
            ml_task=self.ml_task,
            data_source=self.data_source,
            subplot_by=subplot_by,
        )

        relplot_kwargs: dict[str, Any] = {
            "col": col,
            "hue": hue,
            "style": style,
        }

        relplot_kwargs["col_order"] = plot_data[col].unique().tolist() if col else None
        relplot_kwargs["hue_order"] = plot_data[hue].unique().tolist() if hue else None
        relplot_kwargs["style_order"] = (
            plot_data[style].unique().tolist() if style else None
        )

        if style:
            relplot_kwargs["dashes"] = {"train": (5, 5), "test": ""}

        if "cross-validation" in self.report_type:
            relplot_kwargs["units"] = "split"
            relplot_kwargs["alpha"] = 0.4

            # Convert the "split" column from category to string to avoid pandas future
            # warning. See: https://github.com/mwaskom/seaborn/issues/3891
            plot_data["split"] = plot_data["split"].astype(str)

        self.facet_ = sns.relplot(
            data=plot_data,
            kind="line",
            estimator=None,
            x="fpr",
            y="tpr",
            **_validate_style_kwargs(relplot_kwargs, self._default_relplot_kwargs),
        )

        self.figure_, self.ax_ = self.facet_.figure, self.facet_.axes.flatten()

        # Create space under the plot to fit the manually created legends.
        n_legend_rows = plot_data[hue].nunique() if hue else 1
        legend_height_inches = (
            n_legend_rows * 0.25 + 1 + (0.25 if plot_chance_level else 0)
        )
        current_height = self.figure_.get_figheight()
        new_height = current_height + legend_height_inches
        self.figure_.set_figheight(new_height)

        # Build a legend for each subplot.
        for idx, ax in enumerate(self.ax_):
            if plot_chance_level:
                ax.plot((0, 1), (0, 1), **self._default_chance_level_kwargs)
            col_value = (
                relplot_kwargs["col_order"][idx]
                if relplot_kwargs["col_order"]
                else None
            )
            subplot_data = plot_data[plot_data[col] == col_value] if col else plot_data
            _build_custom_legend_with_stats(
                ax=ax,
                subplot_data=subplot_data,
                hue=hue,
                style=style,
                hue_order=relplot_kwargs["hue_order"],
                style_order=relplot_kwargs["style_order"],
                is_cross_validation="cross-validation" in self.report_type,
                statistic_column_name="roc_auc",
                statistic_acronym="AUC",
                chance_level_label="Chance level (AUC = 0.5)",
            )

        if self.ml_task == "binary-classification" and self.pos_label is not None:
            info_pos_label = f"Positive label: {self.pos_label}"
        else:
            info_pos_label = None

        info_data_source = (
            f"Data source: {self.data_source.capitalize()} set"
            if self.data_source in ("train", "test")
            else "Data source: external set"
            if self.data_source == "X_y"
            else None
        )

        title = "ROC Curve"
        if "comparison" not in self.report_type:
            title += f" for {self.roc_curve['estimator'].cat.categories.item()}"
        self.figure_.suptitle(
            "\n".join(filter(None, [title, info_pos_label, info_data_source]))
        )

        for ax in self.ax_:
            ax.set(
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
            )

            if despine:
                _despine_matplotlib_axis(ax)

        if len(self.ax_) == 1:
            self.ax_ = self.ax_[0]

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
                            "estimator": y_true_i.estimator_name,
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
                        "estimator": y_true_i.estimator_name,
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
                                "estimator": y_true_i.estimator_name,
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
                            "estimator": y_true_i.estimator_name,
                            "data_source": y_true_i.data_source,
                            "split": y_true_i.split,
                            "label": class_,
                            "roc_auc": roc_auc_class_i,
                        }
                    )

        dtypes = {
            "estimator": "category",
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

            - `estimator`: Name of the estimator (when comparing estimators)
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
            indexing_columns = ["estimator"]
        else:  # self.report_type == "comparison-cross-validation"
            indexing_columns = ["estimator", "split"]

        if self.data_source == "both":
            indexing_columns += ["data_source"]

        if self.ml_task == "binary-classification":
            columns = indexing_columns + statistical_columns
        else:
            columns = indexing_columns + ["label"] + statistical_columns

        return df[columns]

    # ignore the type signature because we override kwargs by specifying the name of
    # the parameters for the user.
    def set_style(  # type: ignore[override]
        self,
        *,
        policy: Literal["override", "update"] = "update",
        relplot_kwargs: dict | None = None,
        chance_level_kwargs: dict | None = None,
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : {"override", "update"}, default="update"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        relplot_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`seaborn.relplot` for rendering
            the ROC curve(s). Common options include `palette`, `alpha`, `linewidth`,
            etc.

        chance_level_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`matplotlib.pyplot.plot` for
            rendering the chance level line. Common options include `color`,
            `alpha`, `linestyle`, etc.

        Returns
        -------
        self : object
            The instance with a modified style.

        Raises
        ------
        ValueError
            If a style parameter is unknown.
        """
        return super().set_style(
            policy=policy,
            relplot_kwargs=relplot_kwargs or {},
            chance_level_kwargs=chance_level_kwargs or {},
        )
