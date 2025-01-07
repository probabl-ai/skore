"""Define PrimitiveItem.

PrimitiveItems represents a primitive item with creation and update timestamps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from skore.item.item import Item, ItemTypeError

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


class PrimitiveItem(Item):
    """
    A class to represent a primitive item.

    This class encapsulates a primitive value
    along with its creation and update timestamps.
    """

    def __init__(
        self,
        primitive: Primitive,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a PrimitiveItem.

        Parameters
        ----------
        primitive : Primitive
            The primitive value to store.
        created_at : str, optional
            The creation timestamp as ISO format.
        updated_at : str, optional
            The last update timestamp as ISO format.
        """
        super().__init__(created_at, updated_at)

        self.primitive = primitive

    def as_serializable_dict(self):
        """Get a serializable dict from the item.

        Derived class must call their super implementation
        and merge the result with their output.
        """
        d = super().as_serializable_dict()
        d.update(
            {
                "media_type": "text/markdown",
                "value": self.primitive,
            }
        )
        return d

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
            raise ItemTypeError(f"Type '{primitive.__class__}' is not supported.")

        return cls(primitive=primitive)
