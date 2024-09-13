"""NumpyArrayItem.

This module defines the NumpyArrayItem class, which represents a NumPy array item.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import cached_property

import numpy


class NumpyArrayItem:
    """
    A class to represent a NumPy array item.

    This class encapsulates a NumPy array along with its creation and update timestamps.

    Attributes
    ----------
    array_list : list
        The list representation of the NumPy array.
    created_at : str
        The timestamp when the item was created, in ISO format.
    updated_at : str
        The timestamp when the item was last updated, in ISO format.

    Methods
    -------
    array() : numpy.ndarray
        Returns the NumPy array representation of the stored list.
    factory(array: numpy.ndarray) : NumpyArrayItem
        Creates a new NumpyArrayItem instance from a NumPy array.
    """

    def __init__(
        self,
        array_list: list,
        created_at: str,
        updated_at: str,
    ):
        """
        Initialize a NumpyArrayItem.

        Parameters
        ----------
        array_list : list
            The list representation of the NumPy array.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        self.array_list = array_list
        self.created_at = created_at
        self.updated_at = updated_at

    @cached_property
    def array(self) -> numpy.ndarray:
        """
        Convert the stored list to a NumPy array.

        Returns
        -------
        numpy.ndarray
            The NumPy array representation of the stored list.
        """
        return numpy.asarray(self.array_list)

    @property
    def __dict__(self):
        """
        Get a dictionary representation of the object.

        Returns
        -------
        dict
            A dictionary containing the 'array_list' key.
        """
        return {
            "array_list": self.array_list,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def factory(cls, array: numpy.ndarray) -> NumpyArrayItem:
        """
        Create a new NumpyArrayItem instance from a NumPy array.

        Parameters
        ----------
        array : numpy.ndarray
            The NumPy array to store.

        Returns
        -------
        NumpyArrayItem
            A new NumpyArrayItem instance.
        """
        now = datetime.now(tz=UTC).isoformat()
        instance = cls(
            array_list=array.tolist(),
            created_at=now,
            updated_at=now,
        )

        # add array as cached property
        instance.array = array

        return instance
