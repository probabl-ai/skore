"""Project View models."""

from dataclasses import dataclass

# An ordered list of keys to display
Layout = list[str]


@dataclass
class View:
    """
    A view of a Project.

    Examples
    --------
    >>> from skore.persistence.view import View
    >>> View(layout=["a", "b"])
    View(...)
    """

    layout: Layout
