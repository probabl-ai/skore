from json import dumps

from pandas import DataFrame
from pytest import raises
from skore_remote_project.item import PickleItem, SkrubTableReportItem
from skore_remote_project.item.item import ItemTypeError
from skrub import TableReport


class TestSkrubTableReportItem:
    def test_factory(self, monkeypatch):
        item = SkrubTableReportItem.factory(
            TableReport(
                DataFrame(
                    {
                        "a": [1, 2],
                        "b": ["one", "two"],
                        "c": [11.1, 11.1],
                    }
                )
            )
        )

        assert isinstance(item, SkrubTableReportItem)
        assert isinstance(item, PickleItem)

    def test_factory_exception(self):
        with raises(ItemTypeError):
            SkrubTableReportItem.factory(None)

    def test_representation(self, monkeypatch):
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

        assert item.__representation__ == {
            "representation": {
                "media_type": "text/html",
                "value": report.html_snippet(),
            }
        }

        # Ensure representation is JSONable
        dumps(item.__representation__)
