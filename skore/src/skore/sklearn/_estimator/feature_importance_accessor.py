import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor
from skore.utils._accessor import _check_is_regressor_coef_task


class _FeatureImportanceAccessor(_BaseAccessor, DirNamesMixin):
    """Accessor for feature importance related operations.

    You can access this accessor using the `feature_importance` attribute.
    """

    def __init__(self, parent):
        super().__init__(parent)

    @available_if(_check_is_regressor_coef_task)
    def model_weights(self):
        """Retrieve the coefficients of a regression, including the intercept.

        Only works for LinearRegression, Ridge, and Lasso scikit-learn estimators.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_diabetes(return_X_y=True), random_state=0
        ... )
        >>> regressor = Ridge()
        >>> report = EstimatorReport(
        ...     regressor,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> report.feature_importance.model_weights()
                   Coefficient
        Intercept   152.447736
        Feature #0   21.200004
        Feature #1  -60.476431
        Feature #2  302.876805
        Feature #3  179.410255
        Feature #4    8.909560
        Feature #5  -28.807673
        Feature #6 -149.307189
        Feature #7  112.672129
        Feature #8  250.535095
        Feature #9   99.577694
        """
        estimator = (
            self._parent.estimator_.steps[-1][1]
            if isinstance(self._parent.estimator_, Pipeline)
            else self._parent.estimator_
        )

        feature_names = (
            estimator.feature_names_in_
            if hasattr(estimator, "feature_names_in_")
            else [f"Feature #{i}" for i in range(estimator.n_features_in_)]
        )

        intercept = np.atleast_2d(estimator.intercept_)
        coef = np.atleast_2d(estimator.coef_)

        data = np.concatenate([intercept, coef.T])

        df = pd.DataFrame(
            data=data,
            index=["Intercept"] + list(feature_names),
            columns=(
                [f"Target #{i}" for i in range(data.shape[1])]
                if data.shape[1] != 1
                else ["Coefficient"]
            ),
        )

        return df

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _sort_methods_for_help(self, methods):
        """Override sort method for metrics-specific ordering.

        In short, we display the `report_metrics` first and then the `custom_metric`.
        """

        def _sort_key(method):
            name = method[0]
            if name == "custom_metric":
                priority = 1
            elif name == "report_metrics":
                priority = 2
            else:
                priority = 0
            return priority, name

        return sorted(methods, key=_sort_key)

    def _format_method_name(self, name):
        """Override format method for metrics-specific naming."""
        method_name = f"{name}(...)"
        method_name = method_name.ljust(22)
        if name in self._SCORE_OR_LOSS_INFO and self._SCORE_OR_LOSS_INFO[name][
            "icon"
        ] in ("(↗︎)", "(↘︎)"):
            if self._SCORE_OR_LOSS_INFO[name]["icon"] == "(↗︎)":
                method_name += f"[cyan]{self._SCORE_OR_LOSS_INFO[name]['icon']}[/cyan]"
                return method_name.ljust(43)
            else:  # (↘︎)
                method_name += (
                    f"[orange1]{self._SCORE_OR_LOSS_INFO[name]['icon']}[/orange1]"
                )
                return method_name.ljust(49)
        else:
            return method_name.ljust(29)

    def _get_methods_for_help(self):
        """Override to exclude the plot accessor from methods list."""
        methods = super()._get_methods_for_help()
        return [(name, method) for name, method in methods if name != "plot"]

    def _get_help_panel_title(self):
        return "[bold cyan]Available metrics methods[/bold cyan]"

    def _get_help_legend(self):
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )

    def _get_help_tree_title(self):
        return "[bold cyan]report.metrics[/bold cyan]"

    def __repr__(self):
        """Return a string representation using rich."""
        return self._rich_repr(
            class_name="skore.EstimatorReport.metrics",
            help_method_name="report.metrics.help()",
        )
