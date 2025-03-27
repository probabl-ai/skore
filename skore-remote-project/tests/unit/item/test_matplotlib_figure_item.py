from io import BytesIO
from json import dumps

from joblib import dump, load
from matplotlib.pyplot import subplots
from matplotlib.testing.compare import compare_images
from pytest import raises
from skore_remote_project.item import MatplotlibFigureItem
from skore_remote_project.item.item import (
    ItemTypeError,
    b64_str_to_bytes,
    bytes_to_b64_str,
)


class TestMatplotlibFigureItem:
    def test_factory(self, tmp_path):
        figure, ax = subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

        item = MatplotlibFigureItem.factory(figure)

        # matplotlib being not consistent (`xlink:href` are different between two calls)
        # we can't compare figure bytes directly

        figure.savefig(tmp_path / "figure.png")
        with BytesIO(b64_str_to_bytes(item.figure_b64_str)) as stream:
            load(stream).savefig(tmp_path / "item.png")

        assert compare_images(tmp_path / "figure.png", tmp_path / "item.png", 0) is None

    def test_factory_exception(self):
        with raises(ItemTypeError):
            MatplotlibFigureItem.factory(None)

    def test_parameters(self, tmp_path):
        figure, ax = subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

        item = MatplotlibFigureItem.factory(figure)
        item_parameters = item.__parameters__["parameters"]["parameters"]

        # matplotlib being not consistent (`xlink:href` are different between two calls)
        # we can't compare figure bytes directly

        figure.savefig(tmp_path / "figure.png")
        with BytesIO(b64_str_to_bytes(item_parameters["figure_b64_str"])) as stream:
            load(stream).savefig(tmp_path / "item.png")

        assert compare_images(tmp_path / "figure.png", tmp_path / "item.png", 0) is None
        assert list(item_parameters.keys()) == ["figure_b64_str"]

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
