"""SklearnBaseEstimatorItem.

This module defines the SklearnBaseEstimatorItem class,
which represents a scikit-learn BaseEstimator item.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from .item import Item, ItemTypeError

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
        created_at: Union[str, None] = None,
        updated_at: Union[str, None] = None,
        note: Union[str, None] = None,
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
        note : Union[str, None]
            An optional note.
        """
        super().__init__(created_at, updated_at, note)

        self.estimator_html_repr = estimator_html_repr
        self.estimator_skops = estimator_skops
        self.estimator_skops_untrusted_types = estimator_skops_untrusted_types

    @property
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

    @classmethod
    def factory(
        cls,
        estimator: sklearn.base.BaseEstimator,
        /,
        **kwargs,
    ) -> SklearnBaseEstimatorItem:
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

        if not isinstance(estimator, sklearn.base.BaseEstimator):
            raise ItemTypeError(f"Type '{estimator.__class__}' is not supported.")

        # This line is only needed if we know `estimator` has the right type, so we do
        # it after the type check
        import skops.io

        estimator_html_repr = sklearn.utils.estimator_html_repr(estimator)
        estimator_skops = skops.io.dumps(estimator)
        estimator_skops_untrusted_types = skops.io.get_untrusted_types(
            data=estimator_skops
        )

        return cls(
            estimator_html_repr=estimator_html_repr,
            estimator_skops=estimator_skops,
            estimator_skops_untrusted_types=estimator_skops_untrusted_types,
            **kwargs,
        )

    def as_serializable_dict(self):
        """Convert item to a JSON-serializable dict to used by frontend."""
        return super().as_serializable_dict() | {
            "value": self.estimator_html_repr,
            "media_type": "application/vnd.sklearn.estimator+html",
        }
