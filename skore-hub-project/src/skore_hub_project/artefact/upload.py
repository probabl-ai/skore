"""Function definition of the artefact ``upload``."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property, partial
from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from blake3 import blake3 as Blake3
from joblib import dump
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from ..client.client import Client, HUBClient
from ..item.item import bytes_to_b64_str
from .serializer import Serializer

if TYPE_CHECKING:
    from typing import Any, Final

    import httpx

    from .project import Project


Progress = partial(
    Progress,
    TextColumn("[bold cyan blink]Uploading..."),
    BarColumn(
        complete_style="dark_orange",
        finished_style="dark_orange",
        pulse_style="orange1",
    ),
    TextColumn("[orange1]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    transient=True,
)


def upload_chunk(
    filepath: Path,
    client: httpx.Client,
    url: str,
    offset: int,
    length: int,
) -> str:
    """
    Upload a chunk of the serialized object to the artefacts storage.

    Parameters
    ----------
    filepath : ``Path``
        The path of the file containing the serialized object.
    client : ``httpx.Client``
        The client used to upload the chunk to the artefacts storage.
    url : str
        The url used to upload the chunk to the artefacts storage.
    offset : int
        The start of the chunk in the file containing the serialized object.
    length: int
        The length of the chunk in the file containing the serialized object.

    Returns
    -------
    etag : str
        The ETag assigned by the artefacts storage to the chunk, used to acknowledge the
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
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )

        return response.headers["etag"]


# This is both the threshold at which an object is split into several small parts for
# upload, and the size of these small parts.
CHUNK_SIZE: Final[int] = int(1e7)  # ~10mb


def upload(project: Project, o: Any, type: str) -> str:
    """
    Upload an object to the artefacts storage.

    Parameters
    ----------
    project : ``Project``
        The project where to upload the object.
    o : Any
        The object to upload.
    type : str
        The type to associate to object in the artefacts storage.

    Returns
    -------
    checksum : str
        The checksum of the object after upload to the artefacts storage, based on its
        ``joblib`` serialization.

    Notes
    -----
    An object that was already uploaded in its whole will be ignored.
    """
    with (
        Serializer(o) as serializer,
        HUBClient() as hub_client,
        Client() as standard_client,
        ThreadPoolExecutor() as pool,
    ):
        # Ask for upload urls.
        response = hub_client.post(
            url=f"projects/{project.tenant}/{project.name}/artefacts",
            json=[
                {
                    "checksum": serializer.checksum,
                    "content_type": type,
                    "chunk_number": ceil(serializer.size / CHUNK_SIZE),
                }
            ],
        )

        # An empty response means that an artefact with the same checksum already
        # exists. The object doesn't have to be re-uploaded.
        if urls := response.json():
            task_to_chunk_id = {}

            # Upload each chunk of the serialized object to the artefacts storage, using
            # a disk temporary file.
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
                )

                task_to_chunk_id[task] = chunk_id

            try:
                with Progress() as progress:
                    tasks = as_completed(task_to_chunk_id)
                    total = len(task_to_chunk_id)
                    etags = dict(
                        sorted(
                            (
                                task_to_chunk_id[task],
                                task.result(),
                            )
                            for task in progress.track(tasks, total=total)
                        )
                    )
            except BaseException:
                # Cancel all remaining tasks, especially on `KeyboardInterrupt`.
                for task in task_to_chunk_id:
                    task.cancel()

                raise

            # Acknowledge the upload, to let the hub/storage rebuild the whole.
            hub_client.post(
                url=f"projects/{project.tenant}/{project.name}/artefacts/complete",
                json=[
                    {
                        "checksum": serializer.checksum,
                        "etags": etags,
                    }
                ],
            )

    return serializer.checksum
