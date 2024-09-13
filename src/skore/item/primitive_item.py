"""Define PrimitiveItem.

PrimitiveItems represents a primitive item with creation and update timestamps.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union

    Primitive = Union[
        bool,
        float,
        int,
        str,
        list["Primitive"],
        tuple["Primitive"],
        dict[str | int | float, "Primitive"],
    ]


def is_primitive(obj: object) -> bool:
    """Check if the object is a primitive."""
    if isinstance(obj, (bool, float, int, str)):
        return True
    if isinstance(obj, (list, tuple)):
        return all(is_primitive(item) for item in obj)
    if isinstance(obj, dict):
        return all(
            isinstance(k, (bool, float, int, str)) and is_primitive(v)
            for k, v in obj.items()
        )
    return False


class PrimitiveItem:
    """
    A class to represent a primitive item.

    This class encapsulates a primitive value
    along with its creation and update timestamps.

    Attributes
    ----------
    primitive : Primitive
        The primitive value stored in the item.
    created_at : str
        The timestamp when the item was created as ISO format.
    updated_at : str
        The timestamp when the item was last updated as ISO format.

    Methods
    -------
    factory(primitive: Primitive) -> PrimitiveItem
        Create a new PrimitiveItem with the current timestamp.
    """

    def __init__(
        self,
        primitive: Primitive,
        created_at: str,
        updated_at: str,
    ):
        """
        Initialize a PrimitiveItem.

        Parameters
        ----------
        primitive : Primitive
            The primitive value to store.
        created_at : str
            The creation timestamp as ISO format.
        updated_at : str
            The last update timestamp as ISO format.
        """
        self.primitive = primitive
        self.created_at = created_at
        self.updated_at = updated_at

    @classmethod
    def factory(cls, primitive: Primitive) -> PrimitiveItem:
        """
        Create a new PrimitiveItem with the current timestamp.

        Parameters
        ----------
        primitive : Primitive
            The primitive value to store.

        Returns
        -------
        PrimitiveItem
            A new PrimitiveItem instance.
        """
        if not is_primitive(primitive):
            raise ValueError(f"{primitive} is not Primitive.")

        now = datetime.now(tz=UTC).isoformat()
        return cls(
            primitive=primitive,
            created_at=now,
            updated_at=now,
        )
