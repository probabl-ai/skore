"""Batched artifact-upload pipeline used by ``ReportPayload.upload_artifacts``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from joblib import Parallel, delayed

# Re-exported so tests can monkey-patch ``HUBClient`` in this module's
# namespace (see ``monkeypatch_artifact_hub_client`` in tests' conftest).
from ..client.client import HUBClient  # noqa: F401
from .plan import ArtifactPlan

if TYPE_CHECKING:
    from typing import Final

    import httpx

    from ..project.project import Project
    from .artifact import Artifact


def prepare_artifacts(artifacts: list[Artifact]) -> list[ArtifactPlan | None]:
    """Run ``local_plan()`` on every artifact in parallel.

    Returns a list aligned position-by-position with ``artifacts``;
    entries are ``None`` where the artifact has no content (e.g.,
    a confusion-matrix media on a regression report). Callers that
    only want the non-null plans should filter explicitly.
    """
    if not artifacts:
        return []

    plans: list[ArtifactPlan | None] = Parallel(backend="threading")(
        delayed(a.local_plan)() for a in artifacts
    )
    return plans


def request_upload_urls(
    hub_client: httpx.Client,
    project: Project,
    plans: list[ArtifactPlan],
) -> dict[str, list[dict[str, Any]]]:
    """One ``POST /artifacts`` carrying every plan's metadata.

    Returns a mapping from checksum to its list of upload-URL descriptors.
    Checksums whose content was already uploaded on the hub are absent from
    the returned mapping (the backend issues no URL for them).
    """
    if not plans:
        return {}

    body = [
        {
            "checksum": plan.checksum,
            "chunk_number": plan.chunk_count_for(CHUNK_SIZE),
            "content_type": plan.content_type,
        }
        for plan in plans
    ]

    response = hub_client.post(
        url=f"projects/{project.workspace}/{project.name}/artifacts",
        json=body,
    )

    urls: dict[str, list[dict[str, Any]]] = {}
    for entry in response.json():
        urls.setdefault(entry["checksum"], []).append(entry)
    return urls


# This is both the threshold at which a content is split into several small parts for
# upload, and the size of these small parts.
CHUNK_SIZE: Final[int] = int(1e7)  # ~10mb

# Cap on concurrent chunk PUTs to avoid tripping object-storage rate limits.
MAX_PARALLEL_UPLOADS: Final[int] = 10


def _put_one_chunk(
    storage_client: httpx.Client,
    plan: ArtifactPlan,
    chunk_id: int,
    content: bytes,
    url: str,
    content_type: str,
) -> tuple[str, int, str]:
    """PUT one chunk; return ``(checksum, chunk_id, etag)``."""
    response = storage_client.put(
        url=url,
        content=content,
        headers={"Content-Type": content_type},
        timeout=30,
    )
    return (plan.checksum, chunk_id, response.headers["etag"])


def upload_chunks(
    storage_client: httpx.Client,
    plans: list[ArtifactPlan],
    urls_per_checksum: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[int, str]]:
    """PUT every chunk of every plan in parallel.

    Skips any plan that has no entry in ``urls_per_checksum`` (the backend
    reports its checksum as already uploaded).

    Returns ETags grouped by checksum: ``{checksum: {chunk_id: etag}}``.
    """
    tasks = []
    for plan in plans:
        urls = urls_per_checksum.get(plan.checksum)
        if not urls:
            continue

        url_by_chunk = {
            (entry.get("chunk_id") or 1): entry["upload_url"] for entry in urls
        }
        # Per wire convention: streamed single-part uploads use the artifact's
        # true content-type; multipart chunks use octet-stream.
        is_multipart = len(url_by_chunk) > 1
        content_type = "application/octet-stream" if is_multipart else plan.content_type

        for chunk_id, chunk_bytes in plan.iter_chunks(CHUNK_SIZE):
            tasks.append(
                delayed(_put_one_chunk)(
                    storage_client=storage_client,
                    plan=plan,
                    chunk_id=chunk_id,
                    content=chunk_bytes,
                    url=url_by_chunk[chunk_id],
                    content_type=content_type,
                )
            )

    if not tasks:
        return {}

    results = Parallel(backend="threading", n_jobs=MAX_PARALLEL_UPLOADS)(tasks)

    etags: dict[str, dict[int, str]] = {}
    for checksum, chunk_id, etag in results:
        etags.setdefault(checksum, {})[chunk_id] = etag
    return etags


def complete_uploads(
    hub_client: httpx.Client,
    project: Project,
    etags_per_checksum: dict[str, dict[int, str]],
) -> None:
    """One ``POST /artifacts/complete`` for all newly-uploaded artifacts."""
    if not etags_per_checksum:
        return

    hub_client.post(
        url=f"projects/{project.workspace}/{project.name}/artifacts/complete",
        json=[
            {"checksum": checksum, "etags": etags}
            for checksum, etags in etags_per_checksum.items()
        ],
    )
