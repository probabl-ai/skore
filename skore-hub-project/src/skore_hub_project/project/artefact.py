from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from math import ceil
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

from pathlib import Path


class Artefact:
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

    @cached_property
    def progress(self):
        return Progress(
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

    def __upload_chunk(self, client, url, offset, length) -> str:
        with self.filepath.open("rb") as file:
            file.seek(offset)

            response = client.put(
                url=url,
                content=file.read(length),
                headers={"Content-Type": "application/octet-stream"},
                timeout=None,
            )

            return response.headers["etag"]

    def upload(self, project, *, chunk_size: int = int(1e7)):
        # Ask for upload urls if not already uploaded.
        with AuthenticatedClient(raises=True) as client:
            response = client.post(
                url=f"projects/{project.tenant}/{project.name}/artefacts",
                json=[
                    {
                        "checksum": self.checksum,
                        "content_type": "estimator-report-pickle",
                        "chunk_number": ceil(self.size / chunk_size),
                    }
                ],
            )

        if not (urls := response.json()):
            return

        # Upload each chunk of the pickled report.
        with Client() as client, ThreadPoolExecutor() as pool:
            task_to_chunk_id = {}

            for url in urls:
                chunk_id = url["chunk_id"] or 1
                task = pool.submit(
                    self.__upload_chunk,
                    client=client,
                    url=url["upload_url"],
                    offset=((chunk_id - 1) * chunk_size),
                    length=chunk_size,
                )

                task_to_chunk_id[task] = chunk_id

            try:
                with self.progress:
                    etags = dict(
                        sorted(
                            (task_to_chunk_id[task], task.result())
                            for task in self.progress.track(
                                as_completed(task_to_chunk_id),
                                total=len(task_to_chunk_id),
                            )
                        )
                    )
            except BaseException:
                for task in task_to_chunk_id:
                    task.cancel()

                raise

        # Acknowledgement of upload.
        with AuthenticatedClient(raises=True) as client:
            client.post(
                url=f"projects/{project.tenant}/{project.name}/artefacts/complete",
                json=[
                    {
                        "checksum": self.checksum,
                        "etags": etags,
                    }
                ],
            )
