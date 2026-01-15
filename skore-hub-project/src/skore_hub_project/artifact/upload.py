"""Function definition of the artifact ``upload``."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

from skore_hub_project.client.client import Client, HUBClient

if TYPE_CHECKING:
    from typing import Final

    from httpx import Client as httpx_Client

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
    checksum: str,
    filepath: Path,
    content_type: str,
    pool: ThreadPoolExecutor,
) -> None:
    """
    Upload file to the artifacts storage.

    Parameters
    ----------
    project : ``Project``
        The project where to upload the file.
    checksum : str
        The checksum of the file.
    filepath : Path
        The file to upload.
    content_type : str
        The type of file to upload.
    pool : TheadPoolExecutor
        The pool used to execute the `upload_chunk` threads.
    """
    return
    assert filepath.stat().st_size, "`filepath` must not be empty"

    with HUBClient() as hub_client, Client() as standard_client:
        # Ask for upload urls.
        response = hub_client.post(
            url=f"projects/{project.quoted_tenant}/{project.quoted_name}/artifacts",
            json=[
                {
                    "checksum": checksum,
                    "chunk_number": ceil(filepath.stat().st_size / CHUNK_SIZE),
                    "content_type": content_type,
                }
            ],
        )

        urls = response.json()
        task_to_chunk_id = {}

        # Upload each chunk of the file to the artifacts storage.
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
                filepath=filepath,
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
                    "checksum": checksum,
                    "etags": etags,
                }
            ],
        )
