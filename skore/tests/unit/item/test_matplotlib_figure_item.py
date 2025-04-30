import io
import json

import joblib
import pytest
from matplotlib import get_backend
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from matplotlib.testing.compare import compare_images
from skore.persistence.item import ItemTypeError, MatplotlibFigureItem
from skore.persistence.item.matplotlib_figure_item import mpl_backend
from skore.utils import b64_str_to_bytes, bytes_to_b64_str


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
        with io.BytesIO(b64_str_to_bytes(item.figure_b64_str)) as stream:
            joblib.load(stream).savefig(tmp_path / "item.png")

        assert compare_images(tmp_path / "figure.png", tmp_path / "item.png", 0) is None
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            MatplotlibFigureItem.factory(None)

    def test_ensure_jsonable(self):
        figure, ax = subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

        item = MatplotlibFigureItem.factory(figure)
        item_parameters = item.__parameters__

        json.dumps(item_parameters)

    def test_figure(self, tmp_path):
        figure, ax = subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

        with io.BytesIO() as stream:
            joblib.dump(figure, stream)

            figure_bytes = stream.getvalue()
            figure_b64_str = bytes_to_b64_str(figure_bytes)

        item1 = MatplotlibFigureItem.factory(figure)
        item2 = MatplotlibFigureItem(figure_b64_str)

        figure.savefig(tmp_path / "figure.png")
        item1.figure.savefig(tmp_path / "item1.png")
        item2.figure.savefig(tmp_path / "item2.png")

        assert (
            compare_images(tmp_path / "figure.png", tmp_path / "item1.png", 0) is None
        )
        assert (
            compare_images(tmp_path / "figure.png", tmp_path / "item2.png", 0) is None
        )

    def test_backend_switch(self):
        backend = get_backend()
        # hoppefuly PostScript is never the default in tests ^^
        with mpl_backend(backend="ps"):
            assert get_backend() == "ps"
        assert get_backend() == backend

    def test_backend_switch_in_thread(self):
        # as default osx backend could crash the interpreter
        # if it tries to open a window in a background thread
        # test that the figure property is usable in a thread
        backend = get_backend()

        def foo():
            figure, ax = subplots()
            ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
            φ = MatplotlibFigureItem.factory(figure)
            assert isinstance(φ.figure, Figure)

        import threading

        t = threading.Thread(target=foo)
        t.start()
        t.join()

        assert get_backend() == backend
