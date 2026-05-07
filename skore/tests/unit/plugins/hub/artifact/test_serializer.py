from hashlib import blake2b
from pathlib import Path

from skore_hub_project.artifact.serializer import Serializer


class TestSerializer:
    def test_init(self, tmp_path):
        with Serializer(b"1") as serializer:
            assert serializer.filepath.read_bytes() == b"1"

    def test_enter(self, tmp_path):
        with Serializer(b"1"):
            assert len(list(tmp_path.glob("*"))) == 1

    def test_exit(self, tmp_path):
        with Serializer(b"1"):
            ...

        assert len(list(tmp_path.glob("*"))) == 0

    def test_filepath(self, tmp_path):
        with Serializer(b"1") as serializer:
            assert isinstance(serializer.filepath, Path)
            assert serializer.filepath.parent == tmp_path

    def test_checksum(self):
        checksum = f"blake2b-{blake2b(b'1').hexdigest()}"

        with Serializer(b"1") as serializer:
            assert serializer.checksum == checksum

    def test_size(self):
        with Serializer(b"1") as serializer:
            assert serializer.size == len(b"1")
