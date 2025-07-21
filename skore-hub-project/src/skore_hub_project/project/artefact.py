from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property, partial
from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import blake3.blake3 as Blake3
import joblib
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from ..client.api import Client
from ..client.client import AuthenticatedClient
from ..item.item import bytes_to_b64_str

if TYPE_CHECKING:
    ...


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


class Serializer:
    def __init__(self, o):
        with self.filepath.open("wb") as file:
            joblib.dump(o, file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.filepath.unlink(True)

    @cached_property
    def filepath(self) -> Path:
        with NamedTemporaryFile(mode="w+b", delete=False) as file:
            return Path(file.name)

    @cached_property
    def checksum(self):
        hasher = Blake3(max_threads=(1 if self.size < 1e6 else Blake3.AUTO))
        checksum = hasher.update_mmap(self.filepath).digest()

        return f"blake3-{bytes_to_b64_str(checksum)}"

    @cached_property
    def size(self):
        return self.filepath.stat().st_size


def upload_chunk(
    filepath: Path,
    client: httpx.Client,
    url: str,
    offset: int,
    length: int,
) -> str:
    with filepath.open("rb") as file:
        file.seek(offset)

        response = client.put(
            url=url,
            content=file.read(length),
            headers={"Content-Type": "application/octet-stream"},
            timeout=None,
        )

        return response.headers["etag"]


def upload(project: Project, o: Any, type: str, *, chunk_size: int = int(1e7)) -> str:
    with (
        Serializer(o) as serializer,
        AuthenticatedClient(raises=True) as authenticated_client,
        Client() as standard_client,
        ThreadPoolExecutor() as pool,
    ):
        response = authenticated_client.post(
            url=f"projects/{project.tenant}/{project.name}/artefacts",
            json=[
                {
                    "checksum": serializer.checksum,
                    "content_type": type,
                    "chunk_number": ceil(serializer.size / chunk_size),
                }
            ],
        )

        if urls := response.json():
            task_to_chunk_id = {}

            for url in urls:
                chunk_id = url["chunk_id"] or 1
                task = pool.submit(
                    upload_chunk,
                    filepath=serializer.filepath,
                    client=standard_client,
                    url=url["upload_url"],
                    offset=((chunk_id - 1) * chunk_size),
                    length=chunk_size,
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
                for task in task_to_chunk_id:
                    task.cancel()

                raise

            authenticated_client.post(
                url=f"projects/{project.tenant}/{project.name}/artefacts/complete",
                json=[
                    {
                        "checksum": serializer.checksum,
                        "etags": etags,
                    }
                ],
            )

    return serializer.checksum
