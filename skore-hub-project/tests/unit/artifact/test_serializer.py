from pathlib import Path

from blake3 import blake3 as Blake3
from skore_hub_project import bytes_to_b64_str
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
        checksum = Blake3(b"1").digest()
        checksum = f"blake3-{bytes_to_b64_str(checksum)}"

        with Serializer(b"1") as serializer:
            assert serializer.checksum == checksum

    def test_size(self):
        with Serializer(b"1") as serializer:
            assert serializer.size == len(b"1")
