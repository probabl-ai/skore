from collections.abc import Sequence
from typing import Any, Literal, cast

import seaborn as sns
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.utils import (
    _build_custom_legend_with_stats,
    _ClassifierCurveDisplayMixin,
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


class PrecisionRecallCurveDisplay(_ClassifierCurveDisplayMixin, DisplayMixin):
    """Precision Recall visualization.

    An instance of this class should be created by
    `EstimatorReport.metrics.precision_recall()`. You should not create an
    instance of this class directly.

    Parameters
    ----------
    precision_recall : DataFrame
        The precision-recall curve data to display. The columns are

        - `estimator`
        - `split` (may be null)
        - `label`
        - `threshold`
        - `precision`
        - `recall`.

    average_precision : DataFrame
        The average precision data to display. The columns are

        - `estimator`
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
    >>> display.set_style(relplot_kwargs={"palette": "Set2"}).plot()
    """

    _default_relplot_kwargs: dict[str, Any] = {
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

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        subplot_by: Literal["auto", "label", "estimator", "data_source"]
        | None = "auto",
        despine: bool = True,
    ) -> None:
        """Plot visualization.

        Parameters
        ----------
        subplot_by : {"auto", "label", "estimator", "data_source"} or None, \
            default="auto"
            Column to use for creating subplots. Options:

            - "auto": None for EstimatorReport and Cross-Validation Report, \
              "estimator" for ComparisonReport
            - "label": one subplot per class (multiclass only)
            - "estimator": one subplot per estimator (comparison only)
            - "data_source": one subplot per data source (EstimatorReport with both \
                data sources only)
            - None: no subplots (Not available for comparison in multiclass)

        despine : bool, default=True
            Whether to remove the top and right spines from the plot.

        Notes
        -----
        The average precision (cf. :func:`~sklearn.metrics.average_precision_score`)
        in scikit-learn is computed without any interpolation. To be consistent
        with this metric, the precision-recall curve is plotted without any
        interpolation as well (step-wise style).

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
        >>> display.set_style(relplot_kwargs={"palette": "Set2", "alpha": 0.8}).plot()
        """
        return self._plot(subplot_by=subplot_by, despine=despine)

    def _plot_matplotlib(
        self,
        *,
        subplot_by: Literal["auto", "label", "estimator", "data_source"]
        | None = "auto",
        despine: bool = True,
    ) -> None:
        """Matplotlib implementation of the `plot` method."""
        is_cross_validation = self.report_type in (
            "cross-validation",
            "comparison-cross-validation",
        )

        plot_data = self.frame(with_average_precision=True)

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

        if is_cross_validation:
            relplot_kwargs["units"] = "split"
            relplot_kwargs["alpha"] = 0.4

            # Convert the "split" column from category to string to avoid pandas future
            # warning. See: https://github.com/mwaskom/seaborn/issues/3891
            plot_data["split"] = plot_data["split"].astype(str)

        relplot_kwargs["col_order"] = plot_data[col].unique().tolist() if col else None
        relplot_kwargs["hue_order"] = plot_data[hue].unique().tolist() if hue else None
        relplot_kwargs["style_order"] = (
            plot_data[style].unique().tolist() if style else None
        )

        if style:
            relplot_kwargs["dashes"] = {"train": (5, 5), "test": ""}

        facet_grid = sns.relplot(
            data=plot_data,
            kind="line",
            estimator=None,
            x="recall",
            y="precision",
            **_validate_style_kwargs(relplot_kwargs, self._default_relplot_kwargs),
        )

        self.figure_, self.ax_ = facet_grid.figure, facet_grid.axes.flatten()
        self.lines_ = [line for ax in self.ax_ for line in ax.get_lines()]

        # Create space under the plot to fit the manually created legends.
        n_legend_rows = plot_data[hue].nunique() if hue else 1
        legend_height_inches = n_legend_rows * 0.25 + 1
        current_height = self.figure_.get_figheight()
        new_height = current_height + legend_height_inches
        self.figure_.set_figheight(new_height)

        # Build a legend for each subplot.
        for idx, ax in enumerate(self.ax_):
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
                is_cross_validation=is_cross_validation,
                statistic_column_name="average_precision",
                statistic_acronym="AP",
            )

        if self.ml_task == "binary-classification":
            info_pos_label = (
                f"Positive label: {self.pos_label}"
                if self.pos_label is not None
                else None
            )
        else:
            info_pos_label = None

        info_data_source = (
            f"Data source: {self.data_source.capitalize()} set"
            if self.data_source in ("train", "test")
            else "Data source: external set"
            if self.data_source == "X_y"
            else None
        )

        title = "Precision-Recall Curve"
        if self.report_type in ("estimator", "cross-validation"):
            title += f" for {self.precision_recall['estimator'].cat.categories.item()}"
        self.figure_.suptitle(
            "\n".join(filter(None, [title, info_pos_label, info_data_source]))
        )

        if despine:
            for ax in self.ax_:
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
                            "estimator": y_true_i.estimator_name,
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
                        "estimator": y_true_i.estimator_name,
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
                                "estimator": y_true_i.estimator_name,
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
                            "estimator": y_true_i.estimator_name,
                            "data_source": y_true_i.data_source,
                            "split": y_true_i.split,
                            "label": class_,
                            "average_precision": average_precision_class_i,
                        }
                    )

        dtypes = {
            "estimator": "category",
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

            - `estimator`: Name of the estimator (when comparing estimators)
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
        relplot_kwargs: dict[str, Any] | None = None,
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : Literal["override", "update"], default="update"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        relplot_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`seaborn.relplot` for rendering
            the precision-recall curve(s). Common options include `palette`,
            `alpha`, `linewidth`, etc.

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
        )
