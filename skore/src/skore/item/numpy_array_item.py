"""NumpyArrayItem.

This module defines the NumpyArrayItem class, which represents a NumPy array item.
"""

from __future__ import annotations

from functools import cached_property
from json import dumps, loads
from typing import TYPE_CHECKING

from skore.item.item import Item, ItemTypeError

if TYPE_CHECKING:
    import numpy


class NumpyArrayItem(Item):
    """
    A class to represent a NumPy array item.

    This class encapsulates a NumPy array along with its creation and update timestamps.
    """

    def __init__(
        self,
        array_json: str,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a NumpyArrayItem.

        Parameters
        ----------
        array_json : str
            The JSON representation of the array.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.array_json = array_json

    @cached_property
    def array(self) -> numpy.ndarray:
        """
        The numpy array from the persistence.

        Its content can differ from the original array because it has been serialized
        using `json.dumps` function and not pickled, in order to be
        environment-independent.
        """
        import numpy

        return numpy.asarray(loads(self.array_json))

    def as_serializable_dict(self):
        """Get a serializable dict from the item.

        Derived class must call their super implementation
        and merge the result with their output.
        """
        d = super().as_serializable_dict()
        d.update(
            {
                "media_type": "text/markdown",
                "value": self.array.tolist(),
            }
        )
        return d

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
            raise ItemTypeError(f"Type '{array.__class__}' is not supported.")

        return cls(array_json=dumps(array.tolist()))
