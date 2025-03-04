from pytest import raises
from pandas import DataFrame
from skore.hub.item import SkrubTableReportItem
from skore.hub.item.item import ItemTypeError
from skrub import TableReport


class TestSkrubTableReportItem:
    def test_factory(self, monkeypatch):
        monkeypatch.setattr("secrets.token_hex", lambda: "azertyuiop")

        report = TableReport(
            DataFrame(
                {
                    "a": [1, 2],
                    "b": ["one", "two"],
                    "c": [11.1, 11.1],
                }
            )
        )

        item = SkrubTableReportItem.factory(report)

        assert item.media == report.html_snippet()
        assert item.media_type == "text/html"

    def test_factory_exception(self):
        with raises(ItemTypeError):
            SkrubTableReportItem.factory(None)
