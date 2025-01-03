"""SklearnBaseEstimatorItem.

This module defines the SklearnBaseEstimatorItem class,
which represents a scikit-learn BaseEstimator item.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from skore.item.item import Item, ItemTypeError

if TYPE_CHECKING:
    import sklearn.base


class SklearnBaseEstimatorItem(Item):
    """
    A class to represent a scikit-learn BaseEstimator item.

    This class encapsulates a scikit-learn BaseEstimator along with its
    creation and update timestamps.
    """

    def __init__(
        self,
        estimator_html_repr: str,
        estimator_skops: bytes,
        estimator_skops_untrusted_types: list[str],
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a SklearnBaseEstimatorItem.

        Parameters
        ----------
        estimator_html_repr : str
            The HTML representation of the scikit-learn estimator.
        estimator_skops : bytes
            The skops representation of the scikit-learn estimator.
        estimator_skops_untrusted_types : list[str]
            The list of untrusted types in the skops representation.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.estimator_html_repr = estimator_html_repr
        self.estimator_skops = estimator_skops
        self.estimator_skops_untrusted_types = estimator_skops_untrusted_types

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

        return skops.io.loads(
            self.estimator_skops, trusted=self.estimator_skops_untrusted_types
        )

    def as_serializable_dict(self):
        """Get a serializable dict from the item.

        Derived class must call their super implementation
        and merge the result with their output.
        """
        d = super().as_serializable_dict()
        d.update(
            {
                "value": self.estimator_html_repr,
                "media_type": "application/vnd.sklearn.estimator+html",
            }
        )
        return d

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
        import sklearn.base
        import sklearn.utils
        import skops.io

        if not isinstance(estimator, sklearn.base.BaseEstimator):
            raise ItemTypeError(f"Type '{estimator.__class__}' is not supported.")

        estimator_html_repr = sklearn.utils.estimator_html_repr(estimator)
        estimator_skops = skops.io.dumps(estimator)
        estimator_skops_untrusted_types = skops.io.get_untrusted_types(
            data=estimator_skops
        )

        instance = cls(
            estimator_html_repr=estimator_html_repr,
            estimator_skops=estimator_skops,
            estimator_skops_untrusted_types=estimator_skops_untrusted_types,
        )

        # add estimator as cached property
        instance.estimator = estimator

        return instance
