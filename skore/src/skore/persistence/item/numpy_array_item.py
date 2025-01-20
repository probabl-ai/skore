"""NumpyArrayItem.

This module defines the NumpyArrayItem class, which represents a NumPy array item.
"""

from __future__ import annotations

from json import dumps, loads
from typing import TYPE_CHECKING, Union

from .item import Item, ItemTypeError

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
        created_at: Union[str, None] = None,
        updated_at: Union[str, None] = None,
        note: Union[str, None] = None,
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
        note : Union[str, None]
            An optional note.
        """
        super().__init__(created_at, updated_at, note)

        self.array_json = array_json

    @property
    def array(self) -> numpy.ndarray:
        """
        The numpy array from the persistence.

        Its content can differ from the original array because it has been serialized
        using `json.dumps` function and not pickled, in order to be
        environment-independent.
        """
        import numpy

        return numpy.asarray(loads(self.array_json))

    @classmethod
    def factory(cls, array: numpy.ndarray, /, **kwargs) -> NumpyArrayItem:
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

        return cls(array_json=dumps(array.tolist()), **kwargs)

    def as_serializable_dict(self):
        """Convert item to a JSON-serializable dict to used by frontend."""
        return super().as_serializable_dict() | {
            "media_type": "text/markdown",
            "value": self.array.tolist(),
        }
