from io import BytesIO
from json import dumps as json_dumps

import pytest
from joblib import dump as joblib_dump
from matplotlib.pyplot import subplots
from skore_remote_project.item import MatplotlibFigureItem, PickleItem
from skore_remote_project.item.item import ItemTypeError, bytes_to_b64_str


class TestMatplotlibFigureItem:
    def test_factory(self, tmp_path):
        figure, ax = subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

        item = MatplotlibFigureItem.factory(figure)

        assert isinstance(item, MatplotlibFigureItem)
        assert isinstance(item, PickleItem)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            MatplotlibFigureItem.factory(None)

    @pytest.fixture
    def reproducible(self, monkeypatch):
        import matplotlib

        monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")

        try:
            matplotlib.rcParams["svg.hashsalt"] = "<hashsalt>"
            matplotlib.rcParams["svg.id"] = "<id>"
            yield
        finally:
            matplotlib.rcdefaults()

    @pytest.mark.usefixtures("reproducible")
    def test_representation(self, tmp_path):
        figure, ax = subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

        with BytesIO() as stream:
            joblib_dump(figure, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        with BytesIO() as stream:
            figure.savefig(stream, format="svg", bbox_inches="tight")

            svg_bytes = stream.getvalue()
            svg_b64_str = bytes_to_b64_str(svg_bytes)
            representation = {
                "representation": {
                    "media_type": "image/svg+xml;base64",
                    "value": svg_b64_str,
                }
            }

        item1 = MatplotlibFigureItem.factory(figure)
        item2 = MatplotlibFigureItem(pickle_b64_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        json_dumps(item1.__representation__)
        json_dumps(item2.__representation__)
