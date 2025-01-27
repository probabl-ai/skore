import base64
import io

import joblib
import pytest
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from matplotlib.testing.compare import compare_images
from skore.persistence.item import ItemTypeError, MatplotlibFigureItem
from skore.persistence.item.matplotlib_figure_item import mpl_backend


class FakeFigure(Figure):
    def savefig(self, stream, *args, **kwargs):
        stream.write(b"<figure>")


class TestMatplotlibFigureItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    def test_factory(self, mock_nowstr, tmp_path):
        figure, ax = subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

        item = MatplotlibFigureItem.factory(figure)

        # matplotlib being not consistent (`xlink:href` are different between two calls)
        # we can't compare figure bytes directly

        figure.savefig(tmp_path / "figure.png")
        with io.BytesIO(item.figure_bytes) as stream:
            joblib.load(stream).savefig(tmp_path / "item.png")

        assert compare_images(tmp_path / "figure.png", tmp_path / "item.png", 0) is None
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            MatplotlibFigureItem.factory(None)

    def test_figure(self, tmp_path):
        figure, ax = subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

        with io.BytesIO() as stream:
            joblib.dump(figure, stream)

            figure_bytes = stream.getvalue()

        item1 = MatplotlibFigureItem.factory(figure)
        item2 = MatplotlibFigureItem(figure_bytes)

        figure.savefig(tmp_path / "figure.png")
        item1.figure.savefig(tmp_path / "item1.png")
        item2.figure.savefig(tmp_path / "item2.png")

        assert (
            compare_images(tmp_path / "figure.png", tmp_path / "item1.png", 0) is None
        )
        assert (
            compare_images(tmp_path / "figure.png", tmp_path / "item2.png", 0) is None
        )

    def test_as_serializable_dict(self, mock_nowstr):
        figure = FakeFigure()

        with io.BytesIO() as stream:
            figure.savefig(stream, format="svg", bbox_inches="tight")

            figure_bytes = stream.getvalue()
            figure_b64_str = base64.b64encode(figure_bytes).decode()

        item = MatplotlibFigureItem.factory(figure)

        assert item.as_serializable_dict() == {
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "note": None,
            "media_type": "image/svg+xml;base64",
            "value": figure_b64_str,
        }

    def test_backend_switch(self):
        import matplotlib

        backend = matplotlib.get_backend()
        # hoppefuly PostScript is never the default in tests ^^
        with mpl_backend(backend="ps"):
            assert matplotlib.get_backend() == "ps"
        assert matplotlib.get_backend() == backend
