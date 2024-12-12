"""Base class for all items in the project."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from jinja2 import Environment, FileSystemLoader


class ItemTypeError(Exception):
    """Item type exception.

    Exception raised when an attempt is made to convert an object to an Item, but the
    object's type is not supported.
    """


class Item(ABC):
    """
    Abstract base class for all items in the project.

    This class provides a common interface for all items, including
    creation and update timestamps.

    Parameters
    ----------
    created_at : str | None, optional
        The creation timestamp of the item. If None, the current time is used.
    updated_at : str | None, optional
        The last update timestamp of the item. If None, the current time is used.

    Attributes
    ----------
    created_at : str
        The creation timestamp of the item.
    updated_at : str
        The last update timestamp of the item.
    """

    def __init__(
        self,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ):
        now = datetime.now(tz=timezone.utc).isoformat()

        self.created_at = created_at or now
        self.updated_at = updated_at or now

    @classmethod
    @abstractmethod
    def factory(cls) -> Item:
        """
        Create and return a new instance of the Item.

        Returns
        -------
        Item
            A new instance of the Item.
        """

    @cached_property
    def __parameters__(self) -> dict[str, Any]:
        """
        Get the parameters of the Item instance.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the parameters of the Item instance.
        """
        cls = self.__class__
        cls_parameters = inspect.signature(cls).parameters

        return {parameter: getattr(self, parameter) for parameter in cls_parameters}

    def __repr__(self) -> str:
        """Represent the item."""
        return f"{self.__class__.__name__}(...)"

    @abstractmethod
    def to_serializable(self):
        """Va niquer ta mere."""

    def _repr_html_(self):
        """Represent the item in a notebook."""
        item_folder = Path(__file__).resolve().parent
        templates_env = Environment(loader=FileSystemLoader(item_folder))
        template = templates_env.get_template("standalone_widget.html.jinja")

        static_files_path = item_folder.parent / "ui" / "static" / "assets"

        def read_asset_content(path):
            with open(static_files_path / path) as f:
                return f.read()

        script_content = read_asset_content("index.js")
        styles_content = read_asset_content("index.css")

        context = {
            "id": uuid4().hex,
            "item": self.to_serializable(),
            "script": script_content,
            "styles": styles_content,
        }

        w = template.render(**context)
        with open("/Users/rouk1/dev/skore/skore-ui/public/standalone.html", "w") as f:
            f.write(w)
        return w
