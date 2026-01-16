from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, is_classifier
from sklearn.inspection import permutation_importance

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn.types import Aggregate


class PermutationImportanceDisplay(DisplayMixin):
    """Display to inspect feature importance via feature permutation.

    Parameters
    ----------
    scores : pd.DataFrame
        The scores computed after permuting the input features. The columns are:

        - `feature`
        - `label` or `output` (classification vs. regression)
        - `repetition`
        - `metric`
        - `value`

    report_type : {"estimator", "cross-validation", "comparison-estimator", \
            "comparison-cross-validation"}
        Report type from which the display is created.
    """

    def __init__(self, scores: pd.DataFrame):
        self.scores = scores

    @classmethod
    def _compute_data_for_display(
        self,
        *,
        estimator: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        feature_names: list[str],
        metric: str | Callable | list[str] | tuple[str] | dict[str, Callable] | None,
        n_repeats: int,
        max_samples: float,
        n_jobs: int | None,
        seed: int | None,
    ) -> "PermutationImportanceDisplay":
        scores = permutation_importance(
            estimator=estimator,
            X=X,
            y=y,
            scoring=metric,
            n_repeats=n_repeats,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=seed,
        )

        if "importances" in scores:
            # single metric case -> switch to multi-metric case by wrapping in a dict
            # with the name of the metric
            metric_name = metric if isinstance(metric, str) else metric.__name__
            scores = {metric_name: scores}

        df_importances = []
        for metric_name, metric_values in scores.items():
            metric_importances = np.atleast_3d(metric_values["importances"])

            df_metric_importances = []
            for output_index, output_importances in enumerate(
                np.moveaxis(metric_importances, -1, 0)
            ):
                df = pd.DataFrame(
                    output_importances,
                    index=feature_names,
                    columns=range(1, n_repeats + 1),
                ).melt(var_name="repetition")

                if metric_importances.shape[-1] == 1:  # scalar metric
                    df["label"], df["output"] = np.nan, np.nan
                else:
                    if is_classifier(estimator):
                        df["label"] = estimator.classes_[output_index]
                        df["output"] = np.nan
                    else:
                        df["output"], df["label"] = output_index, np.nan

                df["metric"] = metric_name
                df["feature"] = np.tile(feature_names, n_repeats)
                df_metric_importances.append(df)

            df_metric_importances = pd.concat(df_metric_importances, axis="index")
            df_importances.append(df_metric_importances)

        ordered_columns = [
            "feature",
            "label",
            "output",
            "repetition",
            "metric",
            "value",
        ]
        df_importances = pd.concat(df_importances, axis="index")

        return PermutationImportanceDisplay(scores=df_importances[ordered_columns])

    def plot(
        self,
        *,
        subplot_by: Literal["auto", "estimator", "label", "output"] | None = "auto",
    ) -> None:
        pass

    def frame(
        self, *, aggregate: Aggregate | None = None, flat_index: bool = False
    ) -> pd.DataFrame:
        pass

    # ignore the type signature because we override kwargs by specifying the name of
    # the parameters for the user.
    def set_style(  # type: ignore[override]
        self,
        *,
        policy: Literal["override", "update"] = "update",
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : Literal["override", "update"], default="update"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        Returns
        -------
        self : object
            Returns the instance itself.

        Raises
        ------
        ValueError
            If a style parameter is unknown.
        """
        return super().set_style(
            policy=policy,
        )
