from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union

    Primitive = Union[
        str,
        int,
        float,
        bytes,
        list[Primitive],
        tuple[Primitive],
        dict[str | int | float, Primitive],
    ]


class PrimitiveItem:
    def __init__(self, primitive: Primitive, /):
        self.primitive = primitive

    @classmethod
    def factory(cls, primitive: Primitive) -> PrimitiveItem:
        return cls(primitive)
