"""
SklearnBaseEstimatorItem.

This module defines the ``SklearnBaseEstimatorItem`` class used to serialize instances
of ``sklearn.base.BaseEstimator``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .item import (
    Item,
    ItemTypeError,
    b64_str_to_bytes,
    bytes_to_b64_str,
    lazy_is_instance,
)

if TYPE_CHECKING:
    import sklearn.base


class SklearnBaseEstimatorItem(Item):
    """Serialize instances of ``sklearn.base.BaseEstimator``."""

    def __init__(
        self,
        estimator_html_repr: str,
        estimator_skops_b64_str: str,
        estimator_skops_untrusted_types: list[str],
    ):
        """
        Initialize a ``SklearnBaseEstimatorItem``.

        Parameters
        ----------
        estimator_html_repr : str
            The HTML representation of the ``sklearn`` estimator.
        estimator_skops_b64_str : str
            The ``skops`` serialization of the ``sklearn`` estimator, encoded in a
            base64 str.
        estimator_skops_untrusted_types: list[str]
            The list of untrusted types in the ``skops`` serialization.
        """
        self.estimator_html_repr = estimator_html_repr
        self.estimator_skops_b64_str = estimator_skops_b64_str
        self.estimator_skops_untrusted_types = estimator_skops_untrusted_types

    @property
    def __raw__(self) -> sklearn.base.BaseEstimator:
        """Get the value from the ``SklearnBaseEstimatorItem`` instance."""
        import skops.io

        estimator_skops_bytes = b64_str_to_bytes(self.estimator_skops_b64_str)

        return skops.io.loads(
            data=estimator_skops_bytes,
            trusted=self.estimator_skops_untrusted_types,
        )

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``SklearnBaseEstimatorItem`` instance."""
        return {
            "representation": {
                "media_type": "application/vnd.sklearn.estimator+html",
                "value": self.estimator_html_repr,
            }
        }

    @classmethod
    def factory(cls, value: sklearn.base.BaseEstimator, /) -> SklearnBaseEstimatorItem:
        """
        Create a new ``SklearnBaseEstimatorItem``.

        Create a new ``SklearnBaseEstimatorItem`` from an instance of
        ``sklearn.base.BaseEstimator``.

        Parameters
        ----------
        value : ``sklearn.base.BaseEstimator``
            The value to serialize.

        Returns
        -------
        SklearnBaseEstimatorItem
            A new ``SklearnBaseEstimatorItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` is not an instance of ``sklearn.base.BaseEstimator``.
        """
        if not lazy_is_instance(value, "sklearn.base.BaseEstimator"):
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.")

        import sklearn.utils
        import skops.io

        estimator_html_repr = sklearn.utils.estimator_html_repr(value)
        estimator_skops_bytes = skops.io.dumps(value)
        estimator_skops_b64_str = bytes_to_b64_str(estimator_skops_bytes)
        estimator_skops_untrusted_types = skops.io.get_untrusted_types(
            data=estimator_skops_bytes
        )

        return cls(
            estimator_html_repr,
            estimator_skops_b64_str,
            estimator_skops_untrusted_types,
        )
