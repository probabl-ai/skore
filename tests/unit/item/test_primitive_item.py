from datetime import UTC, datetime

import pytest
from skore.item import PrimitiveItem


class TestPrimitiveItem:
    @pytest.mark.parametrize(
        "primitive",
        [
            "a",
            0,
            1.1,
            True,
            [0, 1, 2],
            (0, 1, 2),
            {"a": 0},
        ],
    )
    def test_factory(self, monkeypatch, mock_nowstr, MockDatetime, primitive):
        monkeypatch.setattr("skore.item.primitive_item.datetime", MockDatetime)

        item = PrimitiveItem.factory(primitive)

        assert vars(item) == {
            "primitive": primitive,
            "created_at": mock_nowstr,
            "updated_at": mock_nowstr,
        }
