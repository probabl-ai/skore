from orjson import OPT_NON_STR_KEYS, OPT_SERIALIZE_NUMPY
from orjson import dumps as orjson_dumps
from pandas import Timestamp
from pytest import raises

from skore_hub_project.json import dumps


def test_json_pandas_timestamp():
    timestamp = Timestamp(2000, 1, 1)

    with raises(TypeError, match="Type is not JSON serializable: Timestamp"):
        orjson_dumps(timestamp, option=(OPT_NON_STR_KEYS | OPT_SERIALIZE_NUMPY))

    assert dumps(timestamp) == b'"2000-01-01T00:00:00"'
