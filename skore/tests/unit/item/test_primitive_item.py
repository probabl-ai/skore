import pytest
from skore.persistence.item import ItemTypeError, PrimitiveItem


class TestPrimitiveItem:
    @pytest.mark.parametrize(
        "primitive",
        [
            0,
            1.1,
            True,
            [0, 1, 2],
            (0, 1, 2),
            {"a": 0},
        ],
    )
    def test_factory(self, monkeypatch, mock_nowstr, MockDatetime, primitive):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

        item = PrimitiveItem.factory(primitive)

        assert item.primitive == primitive
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            PrimitiveItem.factory(None)

        with pytest.raises(ItemTypeError):
            PrimitiveItem.factory("<content>")
