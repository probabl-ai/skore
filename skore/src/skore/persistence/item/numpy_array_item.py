"""NumpyArrayItem.

This module defines the NumpyArrayItem class, which represents a NumPy array item.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Union

from skore.utils import b64_str_to_bytes, bytes_to_b64_str

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
        array_b64_str: str,
        created_at: Union[str, None] = None,
        updated_at: Union[str, None] = None,
        note: Union[str, None] = None,
    ):
        """
        Initialize a NumpyArrayItem.

        Parameters
        ----------
        array_b64_str : str
            The raw bytes of the array in the NumPy serialization format, encoded in
            bas64 string.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        note : Union[str, None]
            An optional note.
        """
        super().__init__(created_at, updated_at, note)

        self.array_b64_str = array_b64_str

    @property
    def array(self) -> numpy.ndarray:
        """
        The numpy array from the persistence.

        Its content can differ from the original array because it has been serialized
        using `json.dumps` function and not pickled, in order to be
        environment-independent.
        """
        import numpy

        array_bytes = b64_str_to_bytes(self.array_b64_str)

        with BytesIO(array_bytes) as stream:
            return numpy.load(stream)

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

        with BytesIO() as stream:
            numpy.save(stream, array, allow_pickle=False)

            array_bytes = stream.getvalue()
            array_b64_str = bytes_to_b64_str(array_bytes)
            return cls(array_b64_str=array_b64_str, **kwargs)
