"""JSON functions definition used in the package."""

from datetime import datetime
from typing import Any

from orjson import OPT_NON_STR_KEYS, OPT_SERIALIZE_NUMPY
from orjson import dumps as orjson_dumps


def dumps(obj: Any, /) -> bytes:
    """Serialize Python objects to JSON."""

    def default(obj: Any) -> str:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError

    return orjson_dumps(
        obj,
        default=default,
        option=(OPT_NON_STR_KEYS | OPT_SERIALIZE_NUMPY),
    )
