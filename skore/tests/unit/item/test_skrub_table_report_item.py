import pytest
from pandas import DataFrame
from skore.item import ItemTypeError, SkrubTableReportItem
from skrub import TableReport


class TestSkrubTableReportItem:
    def test_factory(self, monkeypatch, mock_nowstr, MockDatetime):
        monkeypatch.setattr("skore.item.item.datetime", MockDatetime)
        monkeypatch.setattr("secrets.token_hex", lambda: "azertyuiop")

        df = DataFrame(dict(a=[1, 2], b=["one", "two"], c=[11.1, 11.1]))
        report = TableReport(df)
        item = SkrubTableReportItem.factory(report)

        assert item.media_bytes == report.html_snippet().encode()
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            SkrubTableReportItem.factory(None)
