import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor
from skore.sklearn._estimator.report import EstimatorReport
from skore.utils._accessor import _check_has_coef


class _FeatureImportanceAccessor(_BaseAccessor["EstimatorReport"], DirNamesMixin):
    """Accessor for feature importance related operations.

    You can access this accessor using the `feature_importance` attribute.
    """

    def __init__(self, parent: EstimatorReport) -> None:
        super().__init__(parent)

    @available_if(_check_has_coef())
    def coefficients(self) -> pd.DataFrame:
        """Retrieve the coefficients of a linear model, including the intercept.

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
        >>> report.feature_importance.coefficients()
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
            self._parent.estimator_.feature_names_in_
            if hasattr(self._parent.estimator_, "feature_names_in_")
            else [
                f"Feature #{i}" for i in range(self._parent.estimator_.n_features_in_)
            ]
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

    def _format_method_name(self, name: str) -> str:
        return f"{name}(...)".ljust(29)

    def _get_help_panel_title(self) -> str:
        return "[bold cyan]Available feature importance methods[/bold cyan]"

    def _get_help_tree_title(self) -> str:
        return "[bold cyan]report.feature_importance[/bold cyan]"

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(
            class_name="skore.EstimatorReport.feature_importance",
            help_method_name="report.feature_importance.help()",
        )
