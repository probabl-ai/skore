from pytest import raises

from skore._plugins.hub.artifact.plan import ArtifactPlan


class TestArtifactPlan:
    def test_in_memory(self, tmp_path):
        plan = ArtifactPlan(
            checksum="blake2b-abc",
            size=3,
            content_type="text/plain",
            payload=b"abc",
        )
        assert plan.chunk_count == 1
        assert list(plan.iter_chunks(chunk_size=10)) == [(1, b"abc")]

    def test_disk_backed(self, tmp_path):
        path = tmp_path / "data.bin"
        path.write_bytes(b"x" * 25)
        plan = ArtifactPlan(
            checksum="blake2b-xyz",
            size=25,
            content_type="application/octet-stream",
            payload=path,
        )
        assert plan.chunk_count_for(chunk_size=10) == 3
        chunks = list(plan.iter_chunks(chunk_size=10))
        assert [cid for cid, _ in chunks] == [1, 2, 3]
        assert b"".join(c for _, c in chunks) == b"x" * 25

    def test_rejects_negative_size(self):
        with raises(ValueError):
            ArtifactPlan(
                checksum="x",
                size=-1,
                content_type="text/plain",
                payload=b"",
            )
