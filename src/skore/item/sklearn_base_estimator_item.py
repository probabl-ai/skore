"""SklearnBaseEstimatorItem.

This module defines the SklearnBaseEstimatorItem class,
which represents a scikit-learn BaseEstimator item.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sklearn.base

from skore.item.item import Item


class SklearnBaseEstimatorItem(Item):
    """
    A class to represent a scikit-learn BaseEstimator item.

    This class encapsulates a scikit-learn BaseEstimator along with its
    creation and update timestamps.
    """

    def __init__(
        self,
        estimator_skops,
        estimator_html_repr,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a SklearnBaseEstimatorItem.

        Parameters
        ----------
        estimator_skops : Any
            The skops representation of the scikit-learn estimator.
        estimator_html_repr : str
            The HTML representation of the scikit-learn estimator.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.estimator_skops = estimator_skops
        self.estimator_html_repr = estimator_html_repr

    @cached_property
    def estimator(self) -> sklearn.base.BaseEstimator:
        """
        Convert the stored skops object to a scikit-learn BaseEstimator.

        Returns
        -------
        sklearn.base.BaseEstimator
            The scikit-learn BaseEstimator representation of the stored skops object.
        """
        import skops.io

        return skops.io.loads(self.estimator_skops)

    @classmethod
    def factory(cls, estimator: sklearn.base.BaseEstimator) -> SklearnBaseEstimatorItem:
        """
        Create a SklearnBaseEstimatorItem instance from a scikit-learn BaseEstimator.

        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            The scikit-learn BaseEstimator to store.

        Returns
        -------
        SklearnBaseEstimatorItem
            A new SklearnBaseEstimatorItem instance.
        """
        import sklearn.utils
        import skops.io

        instance = cls(
            estimator_skops=skops.io.dumps(estimator),
            estimator_html_repr=sklearn.utils.estimator_html_repr(estimator),
        )

        # add estimator as cached property
        instance.estimator = estimator

        return instance
