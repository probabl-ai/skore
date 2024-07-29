from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any

    from mandr.item.display_type import DisplayType


@dataclass(kw_only=True, frozen=True)
class ItemMetadata:
    display_type: DisplayType
    created_at: datetime
    updated_at: datetime


@dataclass(kw_only=True, frozen=True)
class Item:
    data: Any
    metadata: ItemMetadata
