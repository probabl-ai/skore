import numpy as np
import pytest
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal
from skore.persistence.item import ItemTypeError, PandasDataFrameItem


class TestPandasDataFrameItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            PandasDataFrameItem.factory(None)

    @pytest.mark.order(0)
    def test_factory(self, mock_nowstr):
        dataframe = DataFrame([{"key": "value"}], Index([0], name="myIndex"))

        orient = PandasDataFrameItem.ORIENT
        index_json = dataframe.index.to_frame(index=False).to_json(orient=orient)
        dataframe_json = dataframe.reset_index(drop=True).to_json(orient=orient)

        item = PandasDataFrameItem.factory(dataframe)

        assert item.index_json == index_json
        assert item.dataframe_json == dataframe_json
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    @pytest.mark.order(1)
    def test_dataframe(self, mock_nowstr):
        dataframe = DataFrame([{"key": "value"}], Index([0], name="myIndex"))

        orient = PandasDataFrameItem.ORIENT
        index_json = dataframe.index.to_frame(index=False).to_json(orient=orient)
        dataframe_json = dataframe.reset_index(drop=True).to_json(orient=orient)

        item1 = PandasDataFrameItem.factory(dataframe)
        item2 = PandasDataFrameItem(
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
        item = PandasDataFrameItem.factory(dataframe)

        # NOTE: isinstance would not work because numpy.array is an instance of list
        assert type(item.dataframe["key"].iloc[0]) is list

    @pytest.mark.order(1)
    def test_dataframe_with_integer_columns_name_and_multi_index(self, mock_nowstr):
        dataframe = DataFrame(
            [[">70", "1M", "M", 1], [">70", "2F", "F", 2]],
            MultiIndex.from_arrays(
                [["france", "usa"], ["paris", "nyc"], ["1", "1"]],
                names=("country", "city", "district"),
            ),
        )

        orient = PandasDataFrameItem.ORIENT
        index_json = dataframe.index.to_frame(index=False).to_json(orient=orient)
        dataframe_json = dataframe.reset_index(drop=True).to_json(orient=orient)

        item1 = PandasDataFrameItem.factory(dataframe)
        item2 = PandasDataFrameItem(
            index_json=index_json,
            dataframe_json=dataframe_json,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert_frame_equal(item1.dataframe, dataframe)
        assert_frame_equal(item2.dataframe, dataframe)
