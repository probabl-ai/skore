import datetime
import pathlib

import pytest
from mandr.item import DisplayType


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
        (
            {"fit_time": [1], "score_time": [1], "test_score": 1},
            DisplayType.CROSS_VALIDATION_RESULTS,
        ),
        (
            # Could be DisplayType.CROSS_VALIDATION_RESULTS but missing key "fit_time"
            {"score_time": [1], "test_score": 1},
            DisplayType.ANY,
        ),
    ],
)
def test_infer(x, expected):
    assert DisplayType.infer(x) == expected


def test_enum():
    assert DisplayType("integer") == DisplayType.INTEGER

    with pytest.raises(ValueError, match="invalid' is not a valid DisplayType"):
        DisplayType("invalid")
