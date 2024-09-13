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
            b"a",
            [0, 1, 2],
            (0, 1, 2),
            {"a": 0},
        ],
    )
    def test_factory(self, monkeypatch, primitive):
        now = datetime.now(tz=UTC).isoformat()
        mocked_datetime = monkeypatch.patch("module_to_mock.datetime")
        mocked_datetime.datetime.now.return_value = JAN_31

        item = PrimitiveItem.factory(primitive)

        assert vars(item) == {"primitive": primitive}
