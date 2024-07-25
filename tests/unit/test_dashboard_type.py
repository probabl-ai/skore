import datetime
import pathlib

import pytest
from mandr.dashboard_type import DashboardType


@pytest.mark.parametrize(
    "x,expected",
    [
        (3, DashboardType.INTEGER),
        (None, DashboardType.ANY),
        ((1, "b"), DashboardType.ANY),
        ("hello", DashboardType.MARKDOWN),
        (3.0, DashboardType.NUMBER),
        (False, DashboardType.BOOLEAN),
        ([1, 2], DashboardType.ARRAY),
        (datetime.date(2024, 7, 24), DashboardType.DATE),
        (datetime.datetime(2024, 7, 24), DashboardType.DATETIME),
        (pathlib.PosixPath("./my_file.txt"), DashboardType.FILE),
        ({"a": 1}, DashboardType.ANY),
        (set([1, 2]), DashboardType.ANY),
    ],
)
def test_infer(x, expected):
    assert DashboardType.infer(x) == expected
