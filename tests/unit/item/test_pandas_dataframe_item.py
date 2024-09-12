import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from skore.item import PandasDataFrameItem


class TestPandasDataFrameItem:
    @pytest.mark.order(0)
    def test_factory(self):
        dataframe = DataFrame([{"key": "value"}])
        dataframe_dict = dataframe.to_dict(orient="tight")

        item = PandasDataFrameItem.factory(dataframe)

        assert vars(item) == {"dataframe_dict": dataframe_dict}

    @pytest.mark.order(1)
    def test_dataframe(self):
        dataframe = DataFrame([{"key": "value"}])
        dataframe_dict = dataframe.to_dict(orient="tight")

        item1 = PandasDataFrameItem.factory(dataframe)
        item2 = PandasDataFrameItem(dataframe_dict)

        assert_frame_equal(item1.dataframe, dataframe)
        assert_frame_equal(item2.dataframe, dataframe)
