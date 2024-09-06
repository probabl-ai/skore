import datetime
import pathlib

import pytest
from sklearn.linear_model import LinearRegression
from skore.item import DisplayType


@pytest.mark.parametrize(
    "x,expected",
    [
        (3, DisplayType.INTEGER),
        (None, DisplayType.ANY),
        ((1, "b"), DisplayType.ANY),
        ("hello", DisplayType.MARKDOWN),
        (3.0, DisplayType.NUMBER),
        (False, DisplayType.BOOLEAN),
        ([1, 2], DisplayType.ARRAY),
        (datetime.date(2024, 7, 24), DisplayType.DATE),
        (datetime.datetime(2024, 7, 24), DisplayType.DATETIME),
        (pathlib.PosixPath("./my_file.txt"), DisplayType.FILE),
        ({"a": 1}, DisplayType.ANY),
        (set([1, 2]), DisplayType.ANY),
        (LinearRegression(), DisplayType.SKLEARN_MODEL),
    ],
)
def test_infer(x, expected):
    assert DisplayType.infer(x) == expected


def test_enum():
    assert DisplayType("integer") == DisplayType.INTEGER

    with pytest.raises(ValueError, match="invalid' is not a valid DisplayType"):
        DisplayType("invalid")
