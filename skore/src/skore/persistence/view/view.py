"""Project View models."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class LayoutItem:
    """A layout item.

    A view has items they can be an project item or a note item
    """

    kind: Literal["note", "item"]
    value: str


# An ordered list of keys to display
Layout = list[LayoutItem]


@dataclass
class View:
    """A view of a Project.

    Examples
    --------
    >>> View(layout=[
        {"kind": "note", "value": "# title"},
        {"kind": "item", "value": "item/key"},
    ])
    View(...)
    """

    layout: Layout
