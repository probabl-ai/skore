"""Item class used to store data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any

    from mandr.item.display_type import DisplayType


@dataclass(kw_only=True, frozen=True)
class ItemMetadata:
    """ItemMetadata class used to store metadata."""

    display_type: DisplayType
    created_at: datetime
    updated_at: datetime


@dataclass(kw_only=True, frozen=True)
class Item:
    """Item class used to store data and metadata."""

    data: Any
    metadata: ItemMetadata
