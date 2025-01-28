"""Project View models."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class LayoutItem:
    """A layout item.

    A view has items they can be an porjetc item or a markdown item
    """

    kind: Literal["markdown", "item"]
    value: str


# An ordered list of keys to display
Layout = list[LayoutItem]


@dataclass
class View:
    """A view of a Project.

    Examples
    --------
    >>> View(layout=[
        {"kind": "markdown", "value": "# title"},
        {"kind": "item", "value": "item/key"},
    ])
    View(...)
    """

    layout: Layout
