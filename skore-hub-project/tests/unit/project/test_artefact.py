from pathlib import Path
from urllib.parse import urljoin

from blake3 import blake3 as Blake3
from httpx import Client
from pytest import fixture
from skore_hub_project.item.item import bytes_to_b64_str
from skore_hub_project.project.artefact import Serializer


class FakeClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def request(self, method, url, **kwargs):
        response = super().request(method, urljoin("http://localhost", url), **kwargs)
        response.raise_for_status()

        return response


@fixture(autouse=True)
def monkeypatch_client(monkeypatch):
    monkeypatch.setattr(
        "skore_hub_project.project.artefact.HUBClient",
        FakeClient,
    )


class TestSerialize:
    def test_init(self, tmp_path):
        with Serializer(1) as serializer:
            assert serializer.filepath.read_bytes() == b"\x80\x04K\x01."

    def test_enter(self, tmp_path):
        with Serializer(1):
            assert len(list(tmp_path.glob("*"))) == 1

    def test_exit(self, tmp_path):
        with Serializer(1):
            ...

        assert len(list(tmp_path.glob("*"))) == 0

    def test_filepath(self, tmp_path):
        with Serializer(1) as serializer:
            assert isinstance(serializer.filepath, Path)
            assert serializer.filepath.parent == tmp_path

    def test_checksum(self):
        checksum = Blake3(b"\x80\x04K\x01.").digest()
        checksum = f"blake3-{bytes_to_b64_str(checksum)}"

        with Serializer(1) as serializer:
            assert serializer.checksum == checksum

    def test_size(self):
        with Serializer(1) as serializer:
            assert serializer.size == len(b"\x80\x04K\x01.")
