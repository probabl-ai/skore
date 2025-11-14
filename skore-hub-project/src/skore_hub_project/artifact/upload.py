"""Function definition of the artifact ``upload``."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

from skore_hub_project.client.client import Client, HUBClient

if TYPE_CHECKING:
    from typing import Any, Final

    from httpx import Client as httpx_Client

    from skore_hub_project.artifact.serializer import Serializer
    from skore_hub_project.project.project import Project


def upload_chunk(
    filepath: Path,
    client: httpx_Client,
    url: str,
    offset: int,
    length: int,
    content_type: str,
) -> str:
    """
    Upload a chunk of the serialized content to the artifacts storage.

    Parameters
    ----------
    filepath : ``Path``
        The path of the file containing the serialized content.
    client : ``httpx.Client``
        The client used to upload the chunk to the artifacts storage.
    url : str
        The url used to upload the chunk to the artifacts storage.
    offset : int
        The start of the chunk in the file containing the serialized content.
    length: int
        The length of the chunk in the file containing the serialized content.
    content_type: strategy
        The type of the content to upload.

    Returns
    -------
    etag : str
        The ETag assigned by the artifacts storage to the chunk, used to acknowledge the
        upload.

    Notes
    -----
    This function is in charge of reading its own chunk to reduce RAM footprint.
    """
    with filepath.open("rb") as file:
        file.seek(offset)

        response = client.put(
            url=url,
            content=file.read(length),
            headers={"Content-Type": content_type},
            timeout=30,
        )

        return response.headers["etag"]


# This is both the threshold at which a content is split into several small parts for
# upload, and the size of these small parts.
CHUNK_SIZE: Final[int] = int(1e7)  # ~10mb


def upload(
    project: Project,
    serializer_cls: type[Serializer],
    content: Any,
    content_type: str,
    pool: ThreadPoolExecutor,
    checksums_being_uploaded: set[str],
) -> str:
    """
    Upload content to the artifacts storage.

    Parameters
    ----------
    project : ``Project``
        The project where to upload the content.
    serializer_cls : type[Serializer]
        The class of serializer to use for the content serialization.
    content : Any
        The content to upload.
    content_type : str
        The type of content to upload.
    pool : TheadPoolExecutor
        The pool used to execute the `upload_chunk` threads.
    checksums_being_uploaded : set[str]
        The checksums that are being uploaded by threads.

    Returns
    -------
    checksum : str
        The checksum of the content before upload to the artifacts storage, based on its
        serialization.

    Notes
    -----
    A content that was already uploaded in its whole will be ignored.
    """
    with (
        serializer_cls(content) as serializer,
        HUBClient() as hub_client,
        Client() as standard_client,
    ):
        # Ask for the artifact if it is not already being uploaded by another thread.
        #
        # An non-empty response means that an artifact with the same checksum already
        # exists. The content doesn't have to be re-uploaded.
        if (
            (serializer.checksum not in checksums_being_uploaded)
            and (
                response := hub_client.get(
                    url=f"projects/{project.quoted_tenant}/{project.quoted_name}/artifacts",
                    params={
                        "artifact_checksum": serializer.checksum,
                        "status": "uploaded",
                    },
                )
            )
            and (not response.json())
        ):
            checksums_being_uploaded.add(serializer.checksum)
            serializer()

            # Ask for upload urls.
            response = hub_client.post(
                url=f"projects/{project.quoted_tenant}/{project.quoted_name}/artifacts",
                json=[
                    {
                        "checksum": serializer.checksum,
                        "chunk_number": ceil(serializer.size / CHUNK_SIZE),
                        "content_type": content_type,
                    }
                ],
            )

            urls = response.json()
            task_to_chunk_id = {}

            # Upload each chunk of the serialized content to the artifacts storage,
            # using a disk temporary file.
            #
            # Each task is in charge of reading its own file chunk at runtime, to reduce
            # RAM footprint.
            #
            # Use `threading` over `asyncio` to ensure compatibility with Jupyter
            # notebooks, where the event loop is already running.
            for url in urls:
                chunk_id = url["chunk_id"] or 1
                task = pool.submit(
                    upload_chunk,
                    filepath=serializer.filepath,
                    client=standard_client,
                    url=url["upload_url"],
                    offset=((chunk_id - 1) * CHUNK_SIZE),
                    length=CHUNK_SIZE,
                    content_type=(
                        content_type if len(urls) == 1 else "application/octet-stream"
                    ),
                )

                task_to_chunk_id[task] = chunk_id

            try:
                tasks = as_completed(task_to_chunk_id)
                etags = dict(
                    sorted((task_to_chunk_id[task], task.result()) for task in tasks)
                )
            except BaseException:
                # Cancel all remaining tasks, especially on `KeyboardInterrupt`.
                for task in task_to_chunk_id:
                    task.cancel()

                raise

            # Acknowledge the upload, to let the hub/storage rebuild the whole.
            hub_client.post(
                url=f"projects/{project.quoted_tenant}/{project.quoted_name}/artifacts/complete",
                json=[
                    {
                        "checksum": serializer.checksum,
                        "etags": etags,
                    }
                ],
            )

        return serializer.checksum
