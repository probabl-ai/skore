"""Batched artifact-upload pipeline."""

from __future__ import annotations

from collections import defaultdict
from hashlib import blake2b
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from joblib import Parallel, delayed

# Re-exported so tests can monkey-patch ``HubClient`` in this module's
# namespace (see ``monkeypatch_artifact_hub_client`` in tests' conftest).
from ..client.client import HubClient  # noqa: F401
from .plan import ArtifactPlan

if TYPE_CHECKING:
    from typing import Final

    import httpx

    from .artifact import Artifact

Checksum = str
ETag = str
ChunkId = int

# Both the threshold above which a content is split into chunks, and the size
# of those chunks.
CHUNK_SIZE: Final[int] = int(1e7)  # ~10mb

# Cap on concurrent chunk PUTs to avoid tripping object-storage rate limits.
MAX_PARALLEL_UPLOADS: Final[int] = 10

# Below this threshold we keep content in memory; above, we spool to disk.
SMALL_CONTENT_THRESHOLD: Final[int] = 10 * 1024 * 1024  # 10 MB


def plan_upload(artifact: Artifact) -> ArtifactPlan | None:
    """Convert an Artifact to an upload plan.

    Returns ``None`` if the artifact has no content to upload for this report
    (e.g., a confusion-matrix media on a regression report).
    """
    content = artifact.content_to_upload()

    if content is None:
        return None

    if isinstance(content, str):
        content = content.encode("utf-8")

    payload: bytes | Path = content
    if len(content) > SMALL_CONTENT_THRESHOLD:
        # Spool to a tempfile so the uploader can stream chunks from disk
        # and release the in-memory bytes once we return.
        with NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(content)
            payload = Path(f.name)

    return ArtifactPlan(
        checksum=f"blake2b-{blake2b(content).hexdigest()}",
        size=len(content),
        content_type=artifact.content_type,
        payload=payload,
    )


def request_upload_urls(
    *,
    hub_client: httpx.Client,
    workspace: str,
    project_name: str,
    plans: list[ArtifactPlan],
) -> list[list[dict[str, Any]]]:
    """One ``POST /artifacts`` carrying every plan's metadata.

    Returns the URL entries for each plan, aligned position-by-position with
    ``plans``. Checksums already present on the hub get an empty list (the
    backend issues no URL for them).
    """
    body = [
        {
            "checksum": plan.checksum,
            "chunk_number": plan.chunk_count_for(CHUNK_SIZE),
            "content_type": plan.content_type,
        }
        for plan in plans
    ]

    response = hub_client.post(
        url=f"projects/{workspace}/{project_name}/artifacts",
        json=body,
    )

    entries_by_checksum: dict[Checksum, list[dict[str, Any]]] = defaultdict(list)
    for entry in response.json():
        entries_by_checksum[entry["checksum"]].append(entry)

    return [entries_by_checksum.get(plan.checksum, []) for plan in plans]


def _put_one_chunk(
    *,
    storage_client: httpx.Client,
    checksum: Checksum,
    chunk_id: ChunkId,
    url: str,
    content: bytes,
    content_type: str,
) -> tuple[Checksum, ChunkId, ETag]:
    """PUT one chunk; return ``(checksum, chunk_id, etag)``."""
    response = storage_client.put(
        url=url,
        content=content,
        headers={"Content-Type": content_type},
        timeout=30,
    )
    return (checksum, chunk_id, response.headers["etag"])


def upload_chunks(
    *,
    storage_client: httpx.Client,
    plans: list[ArtifactPlan],
    urls_per_plan: list[list[dict[str, Any]]],
) -> dict[Checksum, dict[ChunkId, ETag]]:
    """PUT every chunk of every plan in parallel.

    Returns ETags grouped by checksum: ``{checksum: {chunk_id: etag}}``.
    """
    chunks_to_upload: list[tuple[Checksum, str, ChunkId, str, bytes]] = []
    for plan, url_entries in zip(plans, urls_per_plan, strict=True):
        if not url_entries:
            continue
        # Wire convention: streamed single-part uploads use the artifact's
        # true content-type; multipart chunks use octet-stream.
        content_type = (
            "application/octet-stream" if len(url_entries) > 1 else plan.content_type
        )
        for url_entry, chunk_bytes in zip(
            url_entries, plan.iter_chunks(CHUNK_SIZE), strict=True
        ):
            chunks_to_upload.append(
                (
                    plan.checksum,
                    content_type,
                    url_entry.get("chunk_id") or 1,
                    url_entry["upload_url"],
                    chunk_bytes,
                )
            )

    if not chunks_to_upload:
        return {}

    results = Parallel(backend="threading", n_jobs=MAX_PARALLEL_UPLOADS)(
        delayed(_put_one_chunk)(
            storage_client=storage_client,
            checksum=checksum,
            content_type=content_type,
            chunk_id=chunk_id,
            url=url,
            content=content,
        )
        for checksum, content_type, chunk_id, url, content in chunks_to_upload
    )

    etags: dict[Checksum, dict[ChunkId, ETag]] = defaultdict(dict)
    for checksum, chunk_id, etag in results:
        etags[checksum][chunk_id] = etag
    return dict(etags)


def complete_uploads(
    *,
    hub_client: httpx.Client,
    workspace: str,
    project_name: str,
    etags_per_checksum: dict[Checksum, dict[ChunkId, ETag]],
) -> None:
    """One ``POST /artifacts/complete`` for all newly-uploaded artifacts."""
    if not etags_per_checksum:
        return

    hub_client.post(
        url=f"projects/{workspace}/{project_name}/artifacts/complete",
        json=[
            {"checksum": checksum, "etags": etags}
            for checksum, etags in etags_per_checksum.items()
        ],
    )


def upload_artifacts(
    *,
    hub_client: httpx.Client,
    storage_client: httpx.Client,
    workspace: str,
    project_name: str,
    artifacts: list[Artifact],
) -> list[ArtifactPlan | None]:
    """Compute plans, upload chunks, return plans aligned with the input.

    Entries are ``None`` for artifacts whose ``content_to_upload`` returns
    ``None`` (e.g., a confusion-matrix media on a regression report).

    Plan computation is sequential: many media artifacts call into matplotlib,
    which is not thread-safe.
    """
    plans = [plan_upload(artifact) for artifact in artifacts]

    plans_to_upload = [plan for plan in plans if plan is not None]
    if plans_to_upload:
        urls_per_plan = request_upload_urls(
            hub_client=hub_client,
            workspace=workspace,
            project_name=project_name,
            plans=plans_to_upload,
        )
        etags_per_checksum = upload_chunks(
            storage_client=storage_client,
            plans=plans_to_upload,
            urls_per_plan=urls_per_plan,
        )
        complete_uploads(
            hub_client=hub_client,
            workspace=workspace,
            project_name=project_name,
            etags_per_checksum=etags_per_checksum,
        )

    return plans
