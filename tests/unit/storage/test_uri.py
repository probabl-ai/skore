import pytest
from mandr.storage import URI


class TestURI:
    def test_init(self):
        root = ("r", "o", "o", "t")

        assert URI("/", "r", "/", "o", "/", "o", "/", "t")._URI__segments == root
        assert URI("/r/o", "/o/t")._URI__segments == root
        assert URI("/r/o/o/t")._URI__segments == root
        assert URI("r/o/o/t")._URI__segments == root

        with pytest.raises(ValueError, match="non-empty"):
            URI("/")

        with pytest.raises(ValueError, match="non-empty"):
            URI("")

    def test_segments(self):
        root = ("r", "o", "o", "t")

        assert URI("/", "r", "/", "o", "/", "o", "/", "t").segments == root
        assert URI("/r/o", "/o/t").segments == root
        assert URI("/r/o/o/t").segments == root
        assert URI("r/o/o/t").segments == root

    def test_parent(self):
        parent = URI("/r/o/o")

        assert URI("/", "r", "/", "o", "/", "o", "/", "t").parent == parent
        assert URI("/r/o", "/o/t").parent == parent
        assert URI("/r/o/o/t").parent == parent
        assert URI("r/o/o/t").parent == parent

        with pytest.raises(ValueError):
            parent = URI("/r").parent

    def test_stem(self):
        assert URI("/", "r", "/", "o", "/", "o", "/", "t").stem == "t"
        assert URI("/r/o", "/o/t").stem == "t"
        assert URI("/r/o/o/t").stem == "t"
        assert URI("r/o/o/t").stem == "t"

    def test_truediv(self):
        root = URI("r", "o", "o", "t")

        assert (URI("/r/o/o") / URI("/t")) == root
        assert (URI("/r/o/o") / URI("t")) == root
        assert (URI("/r/o/o") / "/t") == root
        assert (URI("/r/o/o") / "t") == root

    def test_str(self):
        root = "/r/o/o/t"

        assert str(URI("/", "r", "/", "o", "/", "o", "/", "t")) == root
        assert str(URI("/r/o", "/o/t")) == root
        assert str(URI("/r/o/o/t")) == root
        assert str(URI("r/o/o/t")) == root

    def test_repr(self):
        root = "URI(r,o,o,t)"

        assert repr(URI("/", "r", "/", "o", "/", "o", "/", "t")) == root
        assert repr(URI("/r/o", "/o/t")) == root
        assert repr(URI("/r/o/o/t")) == root
        assert repr(URI("r/o/o/t")) == root

    def test_hash(self):
        root = hash(("r", "o", "o", "t"))

        assert hash(URI("/", "r", "/", "o", "/", "o", "/", "t")) == root
        assert hash(URI("/r/o", "/o/t")) == root
        assert hash(URI("/r/o/o/t")) == root
        assert hash(URI("r/o/o/t")) == root

    def test_len(self):
        assert len(URI("/", "r", "/", "o", "/", "o", "/", "t")) == 4
        assert len(URI("/r/o", "/o/t")) == 4
        assert len(URI("/r/o/o/t")) == 4
        assert len(URI("r/o/o/t")) == 4

    def test_eq(self):
        assert URI("/r") != "/r"
        assert URI("/r") != URI("/r/o/o/t")
        assert (
            URI("/", "r", "/", "o", "/", "o", "/", "t")
            == URI("/r/o", "/o/t")
            == URI("/r/o/o/t")
            == URI("r/o/o/t")
        )

    def test_contains(self):
        assert URI("/r") in URI("/r/o/o/t")
        assert URI("/r/o") in URI("/r/o/o/t")
        assert URI("/r/o/o") in URI("/r/o/o/t")
        assert URI("/r/o/o/t") in URI("/r/o/o/t")
        assert URI("/r/o/o/t/s") not in URI("/r/o/o/t")
        assert URI("/r") not in URI("/root")
