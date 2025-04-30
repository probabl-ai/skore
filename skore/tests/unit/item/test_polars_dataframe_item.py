import numpy as np
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal
from skore.persistence.item import ItemTypeError, PolarsDataFrameItem
from skore.persistence.item.polars_dataframe_item import PolarsToJSONError


class TestPolarsDataFrameItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            PolarsDataFrameItem.factory(None)

    def test_factory(self, mock_nowstr):
        dataframe = DataFrame([{"key": "value"}])

        dataframe_json = dataframe.write_json()

        item = PolarsDataFrameItem.factory(dataframe)

        assert item.dataframe_json == dataframe_json
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_dataframe(self, mock_nowstr):
        dataframe = DataFrame([{"key": "value"}])

        dataframe_json = dataframe.write_json()

        item1 = PolarsDataFrameItem.factory(dataframe)
        item2 = PolarsDataFrameItem(
            dataframe_json=dataframe_json,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert_frame_equal(item1.dataframe, dataframe)
        assert_frame_equal(item2.dataframe, dataframe)

    def test_dataframe_with_complex_object(self, mock_nowstr):
        dataframe = DataFrame([{"key": np.array([1, 2])}])

        with pytest.raises(PolarsToJSONError):
            PolarsDataFrameItem.factory(dataframe)
