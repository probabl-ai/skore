from __future__ import annotations

import sys
from uuid import RFC_4122, UUID

if sys.version_info >= (3, 14):
    from uuid import uuid7 as _generate_uuid7
else:
    from uuid6 import uuid7 as _generate_uuid7


def uuid7() -> UUID:
    """Generate a UUIDv7 as a standard-library UUID object."""
    return UUID(str(_generate_uuid7()))


def normalize_report_id(value: object) -> UUID:
    """Normalize and validate a report ID from persisted state."""
    if isinstance(value, UUID):
        report_id = value
    elif isinstance(value, int) and not isinstance(value, bool):
        report_id = UUID(int=value)
    elif isinstance(value, str):
        report_id = UUID(int=int(value)) if value.isdecimal() else UUID(value)
    else:
        raise TypeError(f"Invalid report ID type: {type(value)!r}")

    if report_id.variant != RFC_4122 or report_id.version not in (4, 7):
        raise ValueError(f"Unsupported report ID: {report_id}")
    return report_id
