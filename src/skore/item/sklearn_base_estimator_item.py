"""SklearnBaseEstimatorItem.

This module defines the SklearnBaseEstimatorItem class,
which represents a scikit-learn BaseEstimator item.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import cached_property

import sklearn
import skops.io


class SklearnBaseEstimatorItem:
    """
    A class to represent a scikit-learn BaseEstimator item.

    This class encapsulates a scikit-learn BaseEstimator along with its
    creation and update timestamps.

    Attributes
    ----------
    estimator_skops : Any
        The skops representation of the scikit-learn estimator.
    estimator_html_repr : str
        The HTML representation of the scikit-learn estimator.
    created_at : str
        The timestamp when the item was created, in ISO format.
    updated_at : str
        The timestamp when the item was last updated, in ISO format.
    """

    def __init__(
        self,
        estimator_skops,
        estimator_html_repr,
        created_at: str,
        updated_at: str,
    ):
        """
        Initialize a SklearnBaseEstimatorItem.

        Parameters
        ----------
        estimator_skops : Any
            The skops representation of the scikit-learn estimator.
        estimator_html_repr : str
            The HTML representation of the scikit-learn estimator.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        self.estimator_skops = estimator_skops
        self.estimator_html_repr = estimator_html_repr
        self.created_at = created_at
        self.updated_at = updated_at

    @cached_property
    def estimator(self) -> sklearn.base.BaseEstimator:
        """
        Convert the stored skops object to a scikit-learn BaseEstimator.

        Returns
        -------
        sklearn.base.BaseEstimator
            The scikit-learn BaseEstimator representation of the stored skops object.
        """
        return skops.io.loads(self.estimator_skops)

    @property
    def __dict__(self):
        """
        Get a dictionary representation of the object.

        Returns
        -------
        dict
            A dictionary containing
            the 'estimator_skops'
            and 'estimator_html_repr' keys.
        """
        return {
            "estimator_skops": self.estimator_skops,
            "estimator_html_repr": self.estimator_html_repr,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

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
        now = datetime.now(tz=UTC).isoformat()
        instance = cls(
            estimator_skops=skops.io.dumps(estimator),
            estimator_html_repr=sklearn.utils.estimator_html_repr(estimator),
            created_at=now,
            updated_at=now,
        )

        # add estimator as cached property
        instance.estimator = estimator

        return instance
