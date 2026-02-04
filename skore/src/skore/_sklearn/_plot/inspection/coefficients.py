from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, is_classifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from skore._sklearn._plot.base import BOXPLOT_STYLE, DisplayMixin
from skore._sklearn._plot.inspection.utils import _decorate_matplotlib_axis
from skore._sklearn.feature_names import _get_feature_names
from skore._sklearn.types import ReportType


class CoefficientsDisplay(DisplayMixin):
    """Display to inspect the coefficients of linear models.

    Parameters
    ----------
    coefficients : DataFrame | list[DataFrame]
        The coefficients data to display. The columns are:

        - `estimator`
        - `split`
        - `feature`
        - `label` or `output` (classification vs. regression)
        - `coefficients`

    report_type : {"estimator", "cross-validation", "comparison-estimator", \
            "comparison-cross-validation"}
        Report type from which the display is created.

    Attributes
    ----------
    ax_ : ndarray of matplotlib Axes
        Array of matplotlib Axes with the different matplotlib axis.

    figure_ : matplotlib Figure
        Figure containing the plot.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import EstimatorReport, train_test_split
    >>> iris = load_iris(as_frame=True)
    >>> X, y = iris.data, iris.target
    >>> y = iris.target_names[y]
    >>> split_data = train_test_split(
    ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
    ... )
    >>> report = EstimatorReport(LogisticRegression(), **split_data)
    >>> display = report.inspection.coefficients()
    >>> display.frame()
                  feature       label  coefficients
    0           Intercept      setosa        9.2...
    1   petal length (cm)      setosa       -2.3...
    2    petal width (cm)      setosa       -0.9...
    3    sepal width (cm)      setosa        0.8...
    4   sepal length (cm)      setosa       -0.4...
    5           Intercept  versicolor        1.7...
    6    petal width (cm)  versicolor       -0.7...
    7   sepal length (cm)  versicolor        0.5...
    8    sepal width (cm)  versicolor       -0.2...
    9   petal length (cm)  versicolor       -0.2...
    10          Intercept   virginica      -11.0...
    11  petal length (cm)   virginica        2.5...
    12   petal width (cm)   virginica        1.7...
    13   sepal width (cm)   virginica       -0.5...
    14  sepal length (cm)   virginica       -0.1...
    """

    _default_barplot_kwargs: dict[str, Any] = {
        "aspect": 2,
        "height": 6,
        "palette": "tab10",
    }
    _default_stripplot_kwargs: dict[str, Any] = {
        "alpha": 0.5,
        "aspect": 2,
        "height": 6,
        "palette": "tab10",
    }
    _default_boxplot_kwargs: dict[str, Any] = {
        "whis": 1e10,
        **BOXPLOT_STYLE,
    }

    def __init__(self, *, coefficients: pd.DataFrame, report_type: ReportType):
        self.coefficients = coefficients
        self.report_type = report_type

    def _select_k_features_in_group(
        self, frame: pd.DataFrame, select_k: int
    ) -> pd.DataFrame:
        coefs = frame.groupby("feature")["coefficients"]

        if "split" in frame:
            # Cross-validation
            scores = coefs.apply(lambda x: x.abs().mean())
        else:
            scores = coefs.first().abs()
        scores = cast(pd.Series, scores)

        if select_k > 0:
            selected_features = scores.nlargest(abs(select_k)).index
        else:
            selected_features = scores.nsmallest(abs(select_k)).index

        return frame[frame["feature"].isin(selected_features)]

    def _sort_features_in_group(
        self,
        frame: pd.DataFrame,
        sorting_order: Literal["descending", "ascending"],
    ) -> pd.DataFrame:
        ascending = sorting_order == "ascending"
        if "split" in frame:
            # Cross-validation
            scores = frame.groupby("feature")["coefficients"].apply(
                lambda x: x.abs().mean()
            )
            scores = cast(pd.Series, scores)
            feature_order = scores.sort_values(ascending=ascending).index
            return frame.set_index("feature").loc[feature_order].reset_index()

        return frame.sort_values(
            by="coefficients",
            key=lambda s: s.abs(),
            ascending=ascending,
        ).reset_index(drop=True)

    def _select_k_features(self, frame: pd.DataFrame, select_k: int) -> pd.DataFrame:
        """Select top-k or bottom-k features based on absolute coefficient values."""
        group_cols = self._get_columns_to_groupby(frame=frame)

        if not group_cols:
            return self._select_k_features_in_group(frame, select_k)

        return pd.concat(
            [
                self._select_k_features_in_group(group, select_k)
                for _, group in frame.groupby(group_cols, observed=True)
            ],
            ignore_index=True,
        )

    def _sort_features(
        self, frame: pd.DataFrame, sorting_order: Literal["descending", "ascending"]
    ) -> pd.DataFrame:
        """Sort features by absolute coefficient values."""
        group_cols = self._get_columns_to_groupby(frame=frame)

        if not group_cols:
            return self._sort_features_in_group(frame, sorting_order=sorting_order)

        return pd.concat(
            [
                self._sort_features_in_group(group, sorting_order=sorting_order)
                for _, group in frame.groupby(group_cols, sort=False, observed=True)
            ],
            ignore_index=True,
        )

    def frame(
        self,
        *,
        include_intercept: bool = True,
        select_k: int | None = None,
        sorting_order: Literal["descending", "ascending", None] = "descending",
    ):
        """Get the coefficients in a dataframe format.

        The returned dataframe is not going to contain constant columns or columns
        containing only NaN values.

        Parameters
        ----------
        include_intercept : bool, default=True
            Whether or not to include the intercept in the dataframe.

        select_k : int, default=None
            Select features based on absolute coefficient values:

            - Positive values: select the `select_k` features with largest absolute
              coefficients
            - Negative values: select the `-select_k` features with smallest absolute
              coefficients

            Selection is performed independently within each group:

            - Single estimator reports: For binary classification or single-output
              regression, selection is global. For multiclass classification or
              multi-output regression, selection is performed independently per
              class/output.
            - Cross-validation reports: Grouping follows the same rules as single
              estimator reports. Within each group, features are ranked by the mean
              absolute coefficient values across folds.
            - Comparison reports: Selection is performed independently per estimator,
              and per class/output if applicable.

        sorting_order : {"descending", "ascending", None}, default="descending"
            Sort features by absolute coefficient values:

            - "descending": largest absolute values first
            - "ascending": smallest absolute values first
            - None: preserve original order

            Can be used independently of `select_k`. Sorting is performed within the
            same groups as selection.

        Returns
        -------
        DataFrame
            Dataframe containing the coefficients of the linear model.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import EstimatorReport, train_test_split
        >>> iris = load_iris(as_frame=True)
        >>> X, y = iris.data, iris.target
        >>> y = iris.target_names[y]
        >>> split_data = train_test_split(
        ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
        ... )
        >>> report = EstimatorReport(LogisticRegression(), **split_data)
        >>> display = report.inspection.coefficients()
        >>> display.frame()
                      feature       label  coefficients
        0           Intercept      setosa        9.2...
        1   petal length (cm)      setosa       -2.3...
        2    petal width (cm)      setosa       -0.9...
        3    sepal width (cm)      setosa        0.8...
        4   sepal length (cm)      setosa       -0.4...
        5           Intercept  versicolor        1.7...
        6    petal width (cm)  versicolor       -0.7...
        7   sepal length (cm)  versicolor        0.5...
        8    sepal width (cm)  versicolor       -0.2...
        9   petal length (cm)  versicolor       -0.2...
        10          Intercept   virginica      -11.0...
        11  petal length (cm)   virginica        2.5...
        12   petal width (cm)   virginica        1.7...
        13   sepal width (cm)   virginica       -0.5...
        14  sepal length (cm)   virginica       -0.1...
        """
        if self.report_type == "estimator":
            columns_to_drop = ["estimator", "split"]
        elif self.report_type == "cross-validation":
            columns_to_drop = ["estimator"]
        elif self.report_type == "comparison-estimator":
            columns_to_drop = ["split"]
        elif self.report_type == "comparison-cross-validation":
            columns_to_drop = []
        else:
            raise TypeError(f"Unexpected report type: {self.report_type!r}")

        if self.coefficients["label"].isna().all():
            # regression problem
            columns_to_drop.append("label")
        if self.coefficients["output"].isna().all():
            # classification problem
            columns_to_drop.append("output")

        coefficients = self.coefficients.drop(columns=columns_to_drop)
        if not include_intercept:
            coefficients = coefficients.query("feature != 'Intercept'")

        if sorting_order is not None:
            coefficients = self._sort_features(coefficients, sorting_order)

        if select_k is not None:
            coefficients = self._select_k_features(coefficients, select_k)

        return coefficients

    @DisplayMixin.style_plot
    def plot(
        self,
        *,
        include_intercept: bool = True,
        subplot_by: Literal["auto", "estimator", "label", "output"] | None = "auto",
        select_k: int | None = None,
        sorting_order: Literal["descending", "ascending", None] = "descending",
    ) -> None:
        """Plot the coefficients for the different features.

        Parameters
        ----------
        include_intercept : bool, default=True
            Whether or not to include the intercept in the dataframe.

        subplot_by : {"auto", "estimator", "label", "output"} or None, default="auto"
            The column to use for subplotting and dividing the coefficients into
            subplots. If "auto", not subplotting is performed apart from:

            - when comparing estimators in a multiclass classification or multi-output
              regression problem;
            - when comparing estimators for which the input features are different.

        select_k : int, default=None
            Select features based on absolute coefficient values:

            - Positive values: select the `select_k` features with largest absolute
              coefficients
            - Negative values: select the `-select_k` features with smallest absolute
              coefficients

            Selection is performed independently within each group as described in
            the `frame` method.

        sorting_order : {"descending", "ascending", None}, default="descending"
            Sort features by absolute coefficient values:

            - "descending": largest absolute values first
            - "ascending": smallest absolute values first
            - None: preserve original order

            Can be used independently of `select_k`. Sorting is performed within the
            same groups as selection.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import EstimatorReport, train_test_split
        >>> iris = load_iris(as_frame=True)
        >>> X, y = iris.data, iris.target
        >>> y = iris.target_names[y]
        >>> split_data = train_test_split(
        ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
        ... )
        >>> report = EstimatorReport(LogisticRegression(), **split_data)
        >>> display = report.inspection.coefficients()
        >>> display.plot()
        """
        return self._plot(
            include_intercept=include_intercept,
            subplot_by=subplot_by,
            select_k=select_k,
            sorting_order=sorting_order,
        )

    def _plot_matplotlib(
        self,
        *,
        include_intercept: bool = True,
        subplot_by: Literal["estimator", "label", "output"] | None = None,
        select_k: int | None = None,
        sorting_order: Literal["descending", "ascending", None] = "descending",
    ) -> None:
        """Dispatch the plotting function for matplotlib backend."""
        frame = self.frame(
            include_intercept=include_intercept,
            select_k=select_k,
            sorting_order=sorting_order,
        )

        # Make copy of the dictionary since we are going to pop some keys later
        barplot_kwargs = self._default_barplot_kwargs.copy()
        boxplot_kwargs = self._default_boxplot_kwargs.copy()
        stripplot_kwargs = self._default_stripplot_kwargs.copy()

        if "comparison" in self.report_type:
            return self._plot_comparison(
                frame=frame,
                report_type=self.report_type,
                subplot_by=subplot_by,
                barplot_kwargs=barplot_kwargs,
                boxplot_kwargs=boxplot_kwargs,
                stripplot_kwargs=stripplot_kwargs,
            )
        # EstimatorReport or CrossValidationReport
        return self._plot_single_estimator(
            frame=frame,
            estimator_name=self.coefficients["estimator"][0],
            report_type=self.report_type,
            subplot_by=subplot_by,
            barplot_kwargs=barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
        )

    @staticmethod
    def _get_columns_to_groupby(*, frame: pd.DataFrame) -> list[str]:
        """Get the available columns from which to group by."""
        columns_to_groupby = list[str]()
        if "estimator" in frame.columns:
            columns_to_groupby.append("estimator")
        if "label" in frame.columns:
            columns_to_groupby.append("label")
        if "output" in frame.columns:
            columns_to_groupby.append("output")
        return columns_to_groupby

    def _categorical_plot(
        self,
        *,
        frame: pd.DataFrame,
        report_type: ReportType,
        hue: str | None = None,
        col: str | None = None,
        barplot_kwargs: dict[str, Any] | None = None,
        boxplot_kwargs: dict[str, Any] | None = None,
        stripplot_kwargs: dict[str, Any] | None = None,
    ):
        if "estimator" in report_type:
            facet = sns.catplot(
                data=frame,
                x="coefficients",
                y="feature",
                hue=hue,
                col=col,
                kind="bar",
                **barplot_kwargs,
            )
        else:  # "cross-validation" in report_type
            facet = sns.catplot(
                data=frame,
                x="coefficients",
                y="feature",
                hue=hue,
                col=col,
                kind="strip",
                dodge=True,
                **stripplot_kwargs,
            ).map_dataframe(
                sns.boxplot,
                x="coefficients",
                y="feature",
                hue=hue,
                dodge=True,
                **boxplot_kwargs,
            )
        add_background_features = hue is not None

        self.figure_, self.ax_ = facet.figure, facet.axes.squeeze()
        for ax in self.ax_.flatten():
            _decorate_matplotlib_axis(
                ax=ax,
                add_background_features=add_background_features,
                n_features=frame["feature"].nunique(),
                xlabel="Magnitude of coefficient",
                ylabel="",
            )
        if len(self.ax_.flatten()) == 1:
            self.ax_ = self.ax_.flatten()[0]

    def _plot_single_estimator(
        self,
        *,
        frame: pd.DataFrame,
        estimator_name: str,
        report_type: ReportType,
        subplot_by: Literal["auto", "estimator", "label", "output"] | None,
        barplot_kwargs: dict[str, Any],
        boxplot_kwargs: dict[str, Any],
        stripplot_kwargs: dict[str, Any],
    ) -> None:
        """Plot the coefficients for an `EstimatorReport` or a `CrossValidationReport`.

        An `EstimatorReport` will use a bar plot while a `CrossValidationReport` will
        use a box plot.

        Parameters
        ----------
        frame : pd.DataFrame
            The frame to plot.
        estimator_name : str
            The name of the estimator to plot.
        report_type : {"estimator", "cross-validation"}
            The type of report to plot.
        subplot_by : {"auto", "estimator", "label", "output"} or None
            The column to use for subplotting and dividing the coefficients into
            subplots.
        barplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.barplot` for
            rendering the coefficients with an :class:`~skore.EstimatorReport` or
            :class:`~skore.ComparisonReport` of :class:`~skore.EstimatorReport`.
        boxplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.boxplot` for
            rendering the coefficients with a :class:`~skore.CrossValidationReport` or
            :class:`~skore.ComparisonReport` of :class:`~skore.CrossValidationReport`.
        stripplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.stripplot` for
            rendering the coefficients with a :class:`~skore.CrossValidationReport` or
            :class:`~skore.ComparisonReport` of :class:`~skore.CrossValidationReport`.
        """
        # {"label"} or {"output"} or {}
        columns_to_groupby = self._get_columns_to_groupby(frame=frame)
        if subplot_by == "auto":
            subplot_by = None

        if subplot_by is not None and not len(columns_to_groupby):
            raise ValueError(
                "No columns to group by. `subplot_by` is expected to be None or 'auto'."
            )
        elif subplot_by is not None and subplot_by not in columns_to_groupby:
            raise ValueError(
                f"Column {subplot_by} not found in the frame. It should be one "
                f"of {', '.join(columns_to_groupby + ['auto', 'None'])}."
            )

        if subplot_by is None:
            hue = None if not len(columns_to_groupby) else columns_to_groupby[0]
            if hue is None:
                barplot_kwargs.pop("palette", None)
                stripplot_kwargs.pop("palette", None)
            col = None
        else:
            hue, col = None, subplot_by
            # we don't need the palette and we are at risk of raising an error or
            # deprecation warning if passing palette without a hue
            barplot_kwargs.pop("palette", None)
            stripplot_kwargs.pop("palette", None)

        self._categorical_plot(
            frame=frame,
            report_type=report_type,
            hue=hue,
            col=col,
            barplot_kwargs=barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
        )

        title = f"Coefficients of {estimator_name}"
        if subplot_by is not None:
            title += f" by {subplot_by}"
        self.figure_.suptitle(title)

    @staticmethod
    def _has_same_features(*, frame: pd.DataFrame) -> bool:
        """Check if the features are the same across all estimators."""
        grouped = {
            name: group["feature"].sort_values().tolist()
            for name, group in frame.groupby("estimator", sort=False)
        }
        _, reference_features = grouped.popitem()
        for group_features in grouped.values():
            if group_features != reference_features:
                return False
        return True

    def _plot_comparison(
        self,
        *,
        frame: pd.DataFrame,
        report_type: ReportType,
        subplot_by: Literal["auto", "estimator", "label", "output"] | None,
        barplot_kwargs: dict[str, Any],
        boxplot_kwargs: dict[str, Any],
        stripplot_kwargs: dict[str, Any],
    ) -> None:
        """Plot the coefficients for a `ComparisonReport`.

        Parameters
        ----------
        frame : pd.DataFrame
            The frame to plot.
        report_type : {"comparison-estimator", "comparison-cross-validation"}
            The type of report to plot.
        subplot_by : {"auto", "estimator", "label", "output"} or None
            The column to use for subplotting and dividing the coefficients into
            subplots. If `None`, an automatic choice is made depending on the type of
            reports at hand.
        barplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.barplot` for
            rendering the coefficients with an :class:`~skore.ComparisonReport` of
            :class:`~skore.EstimatorReport`.
        boxplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.boxplot` for
            rendering the coefficients with a :class:`~skore.ComparisonReport` of
            :class:`~skore.CrossValidationReport`.
        stripplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.stripplot` for
            rendering the coefficients with a :class:`~skore.ComparisonReport` of
            :class:`~skore.CrossValidationReport`.
        """
        # help mypy to understand the following variable types
        hue: str | None = None

        # {"estimator", "label"} or {"estimator", "output"} or {"estimator"}
        columns_to_groupby = self._get_columns_to_groupby(frame=frame)
        if subplot_by not in ("auto", None) and subplot_by not in columns_to_groupby:
            additional_subplot_by = ["auto"]
            if "label" in frame.columns and frame["label"].nunique() > 1:
                additional_subplot_by.append("None")

            raise ValueError(
                f"Column {subplot_by} not found in the frame. It should be one "
                f"of {', '.join(columns_to_groupby + additional_subplot_by)}."
            )
        elif subplot_by is None:
            if "label" in frame.columns:
                n_unique = frame["label"].nunique()
            else:
                n_unique = frame["output"].nunique()
            if n_unique > 1:
                raise ValueError(
                    "There are multiple labels or outputs and `subplot_by` is `None`. "
                    "There is too much information to display on a single plot. "
                    "Please provide a column to group by using `subplot_by`."
                )

        has_same_features = self._has_same_features(frame=frame)
        if (frame.columns.isin(["label", "output"]).any() and subplot_by == "auto") or (
            not has_same_features and subplot_by == "auto"
        ):
            # default fallback on subplots by estimator
            # case 1: multiclass classification or multi-output regression
            # therefore, too many information to display on a single plot, by default
            # group by estimator
            # case 2: features cannot be compared across estimators and we therefore
            # need to subplots by estimator
            subplot_by = "estimator"
        elif subplot_by == "auto":
            subplot_by = None

        if subplot_by is None:
            hue, col = columns_to_groupby[0], None
        else:
            # infer if we should group by another column using hue
            hue_groupby = [col for col in columns_to_groupby if col != subplot_by]
            hue = hue_groupby[0] if len(hue_groupby) else None
            col = subplot_by

            if hue is None:
                barplot_kwargs.pop("palette", None)
                stripplot_kwargs.pop("palette", None)

            if not has_same_features and hue == "estimator":
                raise ValueError(
                    "The estimators have different features and should be plotted on "
                    "different axis using `subplot_by='estimator'`."
                )

        self._categorical_plot(
            frame=frame,
            report_type=report_type,
            hue=hue,
            col=col,
            barplot_kwargs={"sharey": has_same_features} | barplot_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            stripplot_kwargs=stripplot_kwargs,
        )

        title = "Coefficients"
        if subplot_by is not None:
            title += f" by {subplot_by}"
        self.figure_.suptitle(title)

    @classmethod
    def _compute_data_for_display(
        cls,
        *,
        estimators: Sequence[BaseEstimator],
        names: list[str],
        splits: list[int | float],
        report_type: ReportType,
    ) -> "CoefficientsDisplay":
        """Compute the data for the display.

        Parameters
        ----------
        estimators : list of estimator
            The estimators to compute the data for.
        names : list of str
            The names of the estimators.
        splits : list of int or np.nan
            The splits to compute the data for.
        report_type : {"estimator", "cross-validation", "comparison-estimator", \
                "comparison-cross-validation"}
            The type of report to compute the data for.

        Returns
        -------
        CoefficientsDisplay
            The data for the display.
        """
        feature_names, est_names, coefficients, split_indices = [], [], [], []
        for estimator, name, split in zip(estimators, names, splits, strict=True):
            if isinstance(estimator, Pipeline):
                preprocessor, predictor = estimator[:-1], estimator[-1]
            else:
                preprocessor, predictor = None, estimator

            if isinstance(predictor, TransformedTargetRegressor):
                predictor = predictor.regressor_

            coef = np.atleast_2d(predictor.coef_).T
            intercept = np.atleast_2d(predictor.intercept_)
            if coef.shape[1] != intercept.shape[1]:
                # it happens that with `fit_intercept=False` and a multi-output
                # regression problem, intercept is a single float. Thus, we need to
                # repeat it for each output
                intercept = np.repeat(intercept, coef.shape[1], axis=1)

            coefficients.append(np.concatenate([intercept, coef]))

            feat_names = ["Intercept"] + _get_feature_names(
                predictor, transformer=preprocessor, n_features=coef.shape[0]
            )
            feature_names.extend(feat_names)
            est_names.extend([name] * len(feat_names))
            split_indices.extend([split] * len(feat_names))

        index = pd.DataFrame(
            {
                "estimator": est_names,
                "split": split_indices,
                "feature": feature_names,
            }
        )

        if coef.shape[1] == 1:
            # binary or single output regression
            columns, require_melting = ["coefficients"], False
            index["label"], index["output"] = np.nan, np.nan
        else:
            require_melting = True
            if is_classifier(predictor):
                # multi-class classification
                columns, var_name = predictor.classes_.tolist(), "label"
                index["output"] = np.nan
            else:
                # multi-output regression
                columns, var_name = [f"{i}" for i in range(coef.shape[1])], "output"
                index["label"] = np.nan
            id_vars, value_name = index.columns.tolist(), "coefficients"

        coefficients = pd.DataFrame(
            np.concatenate(coefficients, axis=0), columns=columns
        )
        coefficients = pd.concat([index, coefficients], axis=1)
        if require_melting:
            # melt the coefficients and ensure alignment with the label/output, split
            # feature names, and estimator names
            coefficients = coefficients.melt(
                id_vars=id_vars,
                value_vars=columns,
                var_name=var_name,
                value_name=value_name,
            )

        return cls(coefficients=coefficients, report_type=report_type)

    # ignore the type signature because we override kwargs by specifying the name of
    # the parameters for the user.
    def set_style(  # type: ignore[override]
        self,
        *,
        policy: Literal["override", "update"] = "update",
        barplot_kwargs: dict[str, Any] | None = None,
        boxplot_kwargs: dict[str, Any] | None = None,
        stripplot_kwargs: dict[str, Any] | None = None,
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : Literal["override", "update"], default="update"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        barplot_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`seaborn.barplot` for
            rendering the coefficients with an :class:`~skore.EstimatorReport` or
            :class:`~skore.ComparisonReport` of :class:`~skore.EstimatorReport`.

        boxplot_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`seaborn.boxplot` for
            rendering the coefficients with a :class:`~skore.CrossValidationReport` or
            :class:`~skore.ComparisonReport` of :class:`~skore.CrossValidationReport`.

        stripplot_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`seaborn.stripplot` for
            rendering the coefficients with a :class:`~skore.CrossValidationReport` or
            :class:`~skore.ComparisonReport` of :class:`~skore.CrossValidationReport`.

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
            barplot_kwargs=barplot_kwargs or {},
            boxplot_kwargs=boxplot_kwargs or {},
            stripplot_kwargs=stripplot_kwargs or {},
        )
