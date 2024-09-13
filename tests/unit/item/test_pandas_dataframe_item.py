import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from skore.item import PandasDataFrameItem


class TestPandasDataFrameItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.item.item.datetime", MockDatetime)

    @pytest.mark.order(0)
    def test_factory(self, mock_nowstr):
        dataframe = DataFrame([{"key": "value"}])
        dataframe_dict = dataframe.to_dict(orient="tight")

        item = PandasDataFrameItem.factory(dataframe)

        assert item.dataframe_dict == dataframe_dict
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    @pytest.mark.order(1)
    def test_dataframe(self, mock_nowstr):
        dataframe = DataFrame([{"key": "value"}])
        dataframe_dict = dataframe.to_dict(orient="tight")

        item1 = PandasDataFrameItem.factory(dataframe)
        item2 = PandasDataFrameItem(
            dataframe_dict=dataframe_dict,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert_frame_equal(item1.dataframe, dataframe)
        assert_frame_equal(item2.dataframe, dataframe)
