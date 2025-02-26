from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import is_classifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor
from skore.sklearn._estimator.report import EstimatorReport
from skore.utils._accessor import _check_has_coef

DataSource = Literal["test", "train", "X_y"]


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
        Intercept     152.4...
        Feature #0     21.2...
        Feature #1    -60.4...
        Feature #2    302.8...
        Feature #3    179.4...
        Feature #4      8.9...
        Feature #5    -28.8...
        Feature #6   -149.3...
        Feature #7    112.6...
        Feature #8    250.5...
        Feature #9     99.5...
        """
        estimator = self._parent.estimator_

        if isinstance(estimator, Pipeline):
            feature_names = estimator[:-1].get_feature_names_out()
        else:
            if hasattr(estimator, "feature_names_in_"):
                feature_names = estimator.feature_names_in_
            else:
                feature_names = [
                    f"Feature #{i}" for i in range(estimator.n_features_in_)
                ]

        linear_model = estimator[-1] if isinstance(estimator, Pipeline) else estimator
        intercept = np.atleast_2d(linear_model.intercept_)
        coef = np.atleast_2d(linear_model.coef_)

        data = np.concatenate([intercept, coef.T])

        if data.shape[1] == 1:
            columns = ["Coefficient"]
        elif is_classifier(estimator):
            columns = [f"Class #{i}" for i in range(data.shape[1])]
        else:
            columns = [f"Target #{i}" for i in range(data.shape[1])]

        df = pd.DataFrame(
            data=data,
            index=["Intercept"] + list(feature_names),
            columns=columns,
        )

        return df

    def feature_permutation(
        self,
        *,
        data_source: DataSource = "test",
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        scoring=None,
        **kwargs,
    ) -> pd.DataFrame:
        """Report the permutation importance of the estimator."""
        # TODO docs
        return self._feature_permutation(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
            scoring=scoring,
            **kwargs,
        )

    def _feature_permutation(
        self,
        *,
        data_source: DataSource = "test",
        data_source_hash: Optional[int] = None,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        scoring: Union[str, list[str], None] = None,  # Typing TODO
        **kwargs,
    ) -> pd.DataFrame:
        if data_source_hash is None:
            X_, y_true, data_source_hash = self._get_X_y_and_data_source_hash(
                data_source=data_source, X=X, y=y
            )

        # build the cache key components to finally create a tuple that will be used
        # to check if the metric has already been computed
        cache_key_parts: list[Any] = [
            self._parent._hash,
            "permutation_importance",
            data_source,
        ]

        if data_source_hash is not None:
            cache_key_parts.append(data_source_hash)

        # order arguments by key to ensure cache works
        # n_jobs variable should not be in the cache
        for k, v in sorted(kwargs.items()):
            if k == "n_jobs":
                continue
            cache_key_parts.append(v)

        cache_key = tuple(cache_key_parts)

        if cache_key in self._parent._cache:
            score = self._parent._cache[cache_key]
        else:
            sklearn_score = permutation_importance(
                estimator=self._parent.estimator_,
                X=X_,
                y=y_true,
                scoring=scoring,
                **kwargs,
            )
            score = sklearn_score.get("importances")

            feature_names = (
                self._parent.estimator_.feature_names_in_
                if hasattr(self._parent.estimator_, "feature_names_in_")
                else [f"Feature #{i}" for i in range(X_.shape[1])]
            )

            # If there is more than one metric
            if score is None:
                data = np.concatenate(
                    [v["importances"] for v in sklearn_score.values()]
                )
                n_repeats = data.shape[1]
                index = pd.MultiIndex.from_product(
                    [sklearn_score, feature_names], names=("Metric", "Feature")
                )
                columns = pd.Index(
                    (f"Repeat #{i}" for i in range(n_repeats)), name="Repeat"
                )
                score = pd.DataFrame(data=data, index=index, columns=columns)
            else:
                data = score
                n_repeats = data.shape[1]
                index = pd.Index(feature_names, name="Feature")
                columns = pd.Index(
                    (f"Repeat #{i}" for i in range(n_repeats)), name="Repeat"
                )
                score = pd.DataFrame(data=data, index=index, columns=columns)

            # if random_state is None, we do not cache
            if kwargs.get("random_state") is not None:
                self._parent._cache[cache_key] = score

        return score

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
