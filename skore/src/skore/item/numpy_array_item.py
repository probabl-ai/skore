"""NumpyArrayItem.

This module defines the NumpyArrayItem class, which represents a NumPy array item.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy

from skore.item.item import Item


class NumpyArrayItem(Item):
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
        created_at: str | None = None,
        updated_at: str | None = None,
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
        super().__init__(created_at, updated_at)

        self.array_list = array_list

    @cached_property
    def array(self) -> numpy.ndarray:
        """
        Convert the stored list to a NumPy array.

        Returns
        -------
        numpy.ndarray
            The NumPy array representation of the stored list.
        """
        import numpy

        return numpy.asarray(self.array_list)

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
        import numpy

        if not isinstance(array, numpy.ndarray):
            raise TypeError(f"Type '{array.__class__}' is not supported.")

        instance = cls(array_list=array.tolist())

        # add array as cached property
        instance.array = array

        return instance
