from json import dumps
from io import BytesIO
from pytest import fixture, raises
from joblib import dump, load
from matplotlib import get_backend
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from matplotlib.testing.compare import compare_images
from skore.hub.item import MatplotlibFigureItem
from skore.hub.item.item import (
    ItemTypeError,
    Representation,
    b64_str_to_bytes,
    bytes_to_b64_str,
)
from skore.hub.item.matplotlib_figure_item import mpl_backend


class TestMatplotlibFigureItem:
    @fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.hub.item.item.datetime", MockDatetime)

    def test_factory(self, mock_nowstr, tmp_path):
        figure, ax = subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

        item = MatplotlibFigureItem.factory(figure)

        # matplotlib being not consistent (`xlink:href` are different between two calls)
        # we can't compare figure bytes directly

        figure.savefig(tmp_path / "figure.png")
        with BytesIO(b64_str_to_bytes(item.figure_b64_str)) as stream:
            load(stream).savefig(tmp_path / "item.png")

        assert compare_images(tmp_path / "figure.png", tmp_path / "item.png", 0) is None
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr
        assert item.note is None

    def test_factory_exception(self):
        with raises(ItemTypeError):
            MatplotlibFigureItem.factory(None)

    def test_parameters(self, mock_nowstr, tmp_path):
        figure, ax = subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

        item = MatplotlibFigureItem.factory(figure)
        item_parameters = item.__parameters__

        # matplotlib being not consistent (`xlink:href` are different between two calls)
        # we can't compare figure bytes directly

        figure.savefig(tmp_path / "figure.png")
        with BytesIO(b64_str_to_bytes(item_parameters["figure_b64_str"])) as stream:
            load(stream).savefig(tmp_path / "item.png")

        assert compare_images(tmp_path / "figure.png", tmp_path / "item.png", 0) is None
        assert item_parameters["created_at"] == mock_nowstr
        assert item_parameters["updated_at"] == mock_nowstr
        assert item_parameters["note"] is None
        assert list(item_parameters.keys()) == [
            "figure_b64_str",
            "created_at",
            "updated_at",
            "note",
        ]

        # Ensure parameters are JSONable
        dumps(item_parameters)

    def test_raw(self, tmp_path):
        figure, ax = subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

        with BytesIO() as stream:
            dump(figure, stream)

            figure_bytes = stream.getvalue()
            figure_b64_str = bytes_to_b64_str(figure_bytes)

        item1 = MatplotlibFigureItem.factory(figure)
        item2 = MatplotlibFigureItem(figure_b64_str)

        # matplotlib being not consistent (`xlink:href` are different between two calls)
        # we can't compare figure bytes directly

        figure.savefig(tmp_path / "figure.png")
        item1.__raw__.savefig(tmp_path / "item1.png")
        item2.__raw__.savefig(tmp_path / "item2.png")

        assert (
            compare_images(tmp_path / "figure.png", tmp_path / "item1.png", 0) is None
        )
        assert (
            compare_images(tmp_path / "figure.png", tmp_path / "item2.png", 0) is None
        )

    def test_representation(self, tmp_path): ...
