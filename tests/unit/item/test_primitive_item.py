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
    def test_factory(self, primitive):
        item = PrimitiveItem.factory(primitive)

        assert vars(item) == {"primitive": primitive}
