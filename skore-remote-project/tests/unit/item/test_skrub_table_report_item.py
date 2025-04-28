from io import BytesIO
from json import dumps as json_dumps

import pytest
from joblib import dump as joblib_dump
from pandas import DataFrame
from skore_remote_project.item import PickleItem, SkrubTableReportItem
from skore_remote_project.item.item import ItemTypeError, bytes_to_b64_str
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
        with pytest.raises(ItemTypeError):
            SkrubTableReportItem.factory(None)

    @pytest.mark.usefixtures("reproducible")
    def test_representation(self, monkeypatch, now):
        report = TableReport(
            DataFrame(
                {
                    "a": [1, 2],
                    "b": ["one", "two"],
                    "c": [11.1, 11.1],
                }
            )
        )

        with BytesIO() as stream:
            joblib_dump(report, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        representation = {
            "representation": {
                "media_type": "text/html",
                "value": report.html_snippet(),
            }
        }

        item1 = SkrubTableReportItem.factory(report)
        item2 = SkrubTableReportItem(pickle_b64_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        json_dumps(item1.__representation__)
        json_dumps(item2.__representation__)
