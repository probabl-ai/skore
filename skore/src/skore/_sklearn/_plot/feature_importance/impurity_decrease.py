from typing import Any, Literal

import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn._plot.feature_importance.utils import _decorate_matplotlib_axis
from skore._sklearn.feature_names import _get_feature_names
from skore._sklearn.types import ReportType


class ImpurityDecreaseDisplay(DisplayMixin):
    """Display to inspect the mean decrease impurity of tree-based models.

    Parameters
    ----------
    importances : DataFrame
        The importances data to display. The columns are:

        - `estimator`
        - `feature`
        - `importances`

    report_type : {"estimator", "cross-validation", "comparison-estimator", \
            "comparison-cross-validation"}
        Report type from which the display is created.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Matplotlib Axes with the plot.

    facet_ : seaborn FacetGrid
        FacetGrid containing the plot.

    figure_ : matplotlib Figure
        Figure containing the plot.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from skore import EstimatorReport, train_test_split
    >>> iris = load_iris(as_frame=True)
    >>> X, y = iris.data, iris.target
    >>> y = iris.target_names[y]
    >>> split_data = train_test_split(
    ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
    ... )
    >>> report = EstimatorReport(
    ...     RandomForestClassifier(random_state=0), **split_data
    ... )
    >>> display = report.feature_importance.impurity_decrease()
    >>> display.frame()
                    feature  importances
    0  sepal length (cm)     0.1...
    1   sepal width (cm)     0.0...
    2  petal length (cm)     0.4...
    3   petal width (cm)     0.3...
    """

    _default_barplot_kwargs: dict[str, Any] = {}

    def __init__(self, *, importances: pd.DataFrame, report_type: ReportType):
        self.importances = importances
        self.report_type = report_type

    @classmethod
    def _compute_data_for_display(
        cls,
        *,
        estimator: BaseEstimator,
        estimator_name: str,
        report_type: ReportType,
    ) -> "ImpurityDecreaseDisplay":
        """Compute the data for the display.

        Parameters
        ----------
        estimator : BaseEstimator
            The estimator to compute the data for.
        estimator_name : str
            The name of the estimator.
        report_type : {"estimator", "cross-validation", "comparison-estimator", \
                "comparison-cross-validation"}
            The type of report to compute the data for.

        Returns
        -------
        ImpurityDecreaseDisplay
            The data for the display.
        """
        if isinstance(estimator, Pipeline):
            preprocessor, predictor = estimator[:-1], estimator[-1]
        else:
            preprocessor, predictor = None, estimator

        n_features = predictor.feature_importances_.shape[0]
        feature_names = _get_feature_names(
            predictor, transformer=preprocessor, n_features=n_features
        )

        importances = pd.DataFrame(
            {
                "estimator": [estimator_name] * n_features,
                "feature": feature_names,
                "importances": predictor.feature_importances_,
            }
        )

        return cls(importances=importances, report_type=report_type)

    def frame(self) -> pd.DataFrame:
        """Get the mean decrease impurity in a dataframe format.

        The returned dataframe is not going to contain constant columns or columns
        containing only NaN values.

        Returns
        -------
        DataFrame
            Dataframe containing the mean decrease impurity of the tree-based model.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from skore import EstimatorReport, train_test_split
        >>> iris = load_iris(as_frame=True)
        >>> X, y = iris.data, iris.target
        >>> y = iris.target_names[y]
        >>> split_data = train_test_split(
        ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
        ... )
        >>> report = EstimatorReport(
        ...     RandomForestClassifier(random_state=0), **split_data
        ... )
        >>> display = report.feature_importance.impurity_decrease()
        >>> display.frame()
                     feature  importances
        0  sepal length (cm)     0.1...
        1   sepal width (cm)     0.0...
        2  petal length (cm)     0.4...
        3   petal width (cm)     0.3...
        """
        if self.report_type == "estimator":
            columns_to_drop = ["estimator"]
        else:
            raise TypeError(f"Unexpected report type: {self.report_type!r}")

        return self.importances.drop(columns=columns_to_drop)

    @DisplayMixin.style_plot
    def plot(self) -> None:
        """Plot the mean decrease impurity for the different features.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from skore import EstimatorReport, train_test_split
        >>> iris = load_iris(as_frame=True)
        >>> X, y = iris.data, iris.target
        >>> y = iris.target_names[y]
        >>> split_data = train_test_split(
        ...     X=X, y=y, random_state=0, as_dict=True, shuffle=True
        ... )
        >>> report = EstimatorReport(RandomForestClassifier(), **split_data)
        >>> display = report.feature_importance.impurity_decrease()
        >>> display.plot()
        """
        return self._plot()

    def _plot_matplotlib(self) -> None:
        """Dispatch the plotting function for matplotlib backend.

        This method creates a bar plot showing the mean decrease impurity for each
        feature using seaborn's catplot.
        """
        barplot_kwargs = self._default_barplot_kwargs.copy()
        frame = self.frame()

        self._plot_single_estimator(
            frame=frame,
            estimator_name=self.importances["estimator"].unique()[0],
            barplot_kwargs=barplot_kwargs,
        )

    def _plot_single_estimator(
        self,
        *,
        frame: pd.DataFrame,
        estimator_name: str,
        barplot_kwargs: dict[str, Any],
    ) -> None:
        """Plot the mean decrease impurity for an `EstimatorReport`.

        A bar plot is used to display the mean decrease impurity values.

        Parameters
        ----------
        frame : pd.DataFrame
            The frame to plot.
        estimator_name : str
            The name of the estimator to plot.
        barplot_kwargs : dict
            Keyword arguments to be passed to :func:`seaborn.barplot` for
            rendering the mean decrease impurity with an
            :class:`~skore.EstimatorReport`.
        """
        self.facet_ = sns.catplot(
            data=frame,
            x="importances",
            y="feature",
            kind="bar",
            **barplot_kwargs,
        )
        self.figure_, self.ax_ = self.facet_.figure, self.facet_.axes.squeeze()
        self.ax_ = self.ax_[()]  # 0-d array
        _decorate_matplotlib_axis(
            ax=self.ax_,
            add_background_features=False,
            n_features=frame["feature"].nunique(),
            xlabel="Mean decrease impurity",
            ylabel="",
        )
        self.figure_.suptitle(f"Mean decrease impurity of {estimator_name}")

    # ignore the type signature because we override kwargs by specifying the name of
    # the parameters for the user.
    def set_style(  # type: ignore[override]
        self,
        *,
        policy: Literal["override", "update"] = "update",
        barplot_kwargs: dict[str, Any] | None = None,
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : {"override", "update"}, default="update"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        barplot_kwargs : dict, default=None
            Keyword arguments to be passed to :func:`seaborn.barplot` for
            rendering the mean decrease impurity with an
            :class:`~skore.EstimatorReport`.

        Returns
        -------
        self : object
            The instance with a modified style.

        Raises
        ------
        ValueError
            If a style parameter is unknown.
        """
        return super().set_style(policy=policy, barplot_kwargs=barplot_kwargs or {})
