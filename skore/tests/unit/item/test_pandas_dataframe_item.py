import io

import numpy as np
import pyarrow
import pytest
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal
from skore.item import ItemTypeError, PandasDataFrameItem


class TestPandasDataFrameItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            PandasDataFrameItem.factory(None)

    def _dataframe_to_bytes(self, dataframe):
        with io.BytesIO() as dataframe_bytes:
            table = pyarrow.Table.from_pandas(dataframe)
            pyarrow.parquet.write_table(table, dataframe_bytes)
            return dataframe_bytes.getvalue()

    @pytest.mark.order(0)
    def test_factory(self, mock_nowstr):
        dataframe = DataFrame([{"key": "value"}], Index([0], name="myIndex"))

        item = PandasDataFrameItem.factory(dataframe)

        assert item.dataframe_bytes == self._dataframe_to_bytes(dataframe)
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    @pytest.mark.order(1)
    def test_dataframe(self, mock_nowstr):
        dataframe = DataFrame([{"key": "value"}], Index([0], name="myIndex"))

        item1 = PandasDataFrameItem.factory(dataframe)
        item2 = PandasDataFrameItem(
            dataframe_bytes=self._dataframe_to_bytes(dataframe),
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert_frame_equal(item1.dataframe, dataframe)
        assert_frame_equal(item2.dataframe, dataframe)

    @pytest.mark.order(1)
    def test_dataframe_with_complex_object(self, mock_nowstr):
        dataframe = DataFrame([{"key": np.array([1])}], Index([0], name="myIndex"))
        item = PandasDataFrameItem.factory(dataframe)

        assert isinstance(item.dataframe["key"].iloc[0], np.ndarray)

    @pytest.mark.order(1)
    def test_dataframe_with_integer_columns_name_and_multiindex(self, mock_nowstr):
        dataframe = DataFrame(
            [[">70", "1M", "M", 1], [">70", "2F", "F", 2]],
            MultiIndex.from_arrays(
                [["france", "usa"], ["paris", "nyc"], ["1", "1"]],
                names=("country", "city", "district"),
            ),
        )

        item1 = PandasDataFrameItem.factory(dataframe)
        item2 = PandasDataFrameItem(
            dataframe_bytes=self._dataframe_to_bytes(dataframe),
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert_frame_equal(item1.dataframe, dataframe)
        assert_frame_equal(item2.dataframe, dataframe)

    def test_get_serializable_dict(self, mock_nowstr):
        dataframe = DataFrame([{"key": np.array([1])}], Index([0], name="myIndex"))
        item = PandasDataFrameItem.factory(dataframe)
        serializable = item.as_serializable_dict()
        assert serializable == {
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "media_type": "application/vnd.dataframe",
            "value": dataframe.fillna("NaN").to_dict(orient="tight"),
        }
