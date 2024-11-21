import numpy as np
import pytest
from polars import DataFrame, Index, MultiIndex
from polars.testing import assert_frame_equal
from skore.item import ItemTypeError, PolarsDataFrameItem


class TestPolarsDataFrameItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            PolarsDataFrameItem.factory(None)

    @pytest.mark.order(0)
    def test_factory(self, mock_nowstr):
        dataframe = DataFrame([{"key": "value"}], Index([0], name="myIndex"))

        orient = PolarsDataFrameItem.ORIENT
        index_json = dataframe.index.to_frame(index=False).to_json(orient=orient)
        dataframe_json = dataframe.reset_index(drop=True).to_json(orient=orient)

        item = PolarsDataFrameItem.factory(dataframe)

        assert item.index_json == index_json
        assert item.dataframe_json == dataframe_json
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    @pytest.mark.order(1)
    def test_dataframe(self, mock_nowstr):
        dataframe = DataFrame([{"key": "value"}], Index([0], name="myIndex"))

        orient = PolarsDataFrameItem.ORIENT
        index_json = dataframe.index.to_frame(index=False).to_json(orient=orient)
        dataframe_json = dataframe.reset_index(drop=True).to_json(orient=orient)

        item1 = PolarsDataFrameItem.factory(dataframe)
        item2 = PolarsDataFrameItem(
            index_json=index_json,
            dataframe_json=dataframe_json,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert_frame_equal(item1.dataframe, dataframe)
        assert_frame_equal(item2.dataframe, dataframe)

    @pytest.mark.order(1)
    def test_dataframe_with_complex_object(self, mock_nowstr):
        dataframe = DataFrame([{"key": np.array([1])}], Index([0], name="myIndex"))
        item = PolarsDataFrameItem.factory(dataframe)

        assert type(item.dataframe["key"].iloc[0]) is list

    @pytest.mark.order(1)
    def test_dataframe_with_integer_columns_name_and_multiindex(self, mock_nowstr):
        dataframe = DataFrame(
            [[">70", "1M", "M", 1], [">70", "2F", "F", 2]],
            MultiIndex.from_arrays(
                [["france", "usa"], ["paris", "nyc"], ["1", "1"]],
                names=("country", "city", "district"),
            ),
        )

        orient = PolarsDataFrameItem.ORIENT
        index_json = dataframe.index.to_frame(index=False).to_json(orient=orient)
        dataframe_json = dataframe.reset_index(drop=True).to_json(orient=orient)

        item1 = PolarsDataFrameItem.factory(dataframe)
        item2 = PolarsDataFrameItem(
            index_json=index_json,
            dataframe_json=dataframe_json,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert_frame_equal(item1.dataframe, dataframe)
        assert_frame_equal(item2.dataframe, dataframe)
