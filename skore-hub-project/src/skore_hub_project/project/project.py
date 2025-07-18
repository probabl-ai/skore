"""Class definition of the ``skore`` hub project."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from functools import cached_property
from math import ceil
from operator import itemgetter
from tempfile import NamedTemporaryFile, TemporaryFile
from types import SimpleNamespace
from typing import TYPE_CHECKING

import blake3.blake3 as Blake3
import joblib
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from ..client.api import Client
from ..client.client import AuthenticatedClient, HTTPStatusError
from ..item import skore_estimator_report_item
from ..item.item import bytes_to_b64_str

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import TypedDict

    from skore import EstimatorReport
    from skore._sklearn._base import _BaseReport

    class Metadata(TypedDict):  # noqa: D101
        id: str
        run_id: str
        key: str
        date: str
        learner: str
        dataset: str
        ml_task: str
        rmse: float | None
        log_loss: float | None
        roc_auc: float | None
        fit_time: float
        predict_time: float


@contextmanager
def dumps(report: _BaseReport) -> Generator[tuple[str, str, int]]:
    """
    Dump a ``report`` using ``joblib`` in a temporary file.

    Returns
    -------
    str
        The temporary filename containing the pickled report.
    str
        The checksum of the pickled report, based on BLAKE3.
    int
        The size of the pickled report.

    Notes
    -----
    The report is pickled without its cache, to avoid salting the checksum.
    The report is pickled on disk to reduce RAM footprint.
    """
    with NamedTemporaryFile(mode="w+b", delete=False) as file:
        filename = file.name

        try:
            cache = report._cache
            report._cache = {}

            try:
                joblib.dump(report, file)
            finally:
                report._cache = cache

            size = file.tell()

            # "First do `f.flush()`, and then do `os.fsync(f.fileno())`, to ensure that
            # all internal buffers associated with f are written to disk."
            #
            # https://docs.python.org/3.13/library/os.html#os.fsync
            file.flush()
            os.fsync(file.fileno())
            file.close()

            # Define if hasher must be on one or more threads: "Note that this can be
            # slower for inputs shorter than ~1 MB"
            #
            # https://github.com/oconnor663/blake3-py
            hasher = Blake3(max_threads=(1 if size < 1e6 else Blake3.AUTO))
            checksum = hasher.update_mmap(filename).digest()

            yield filename, f"blake3-{bytes_to_b64_str(checksum)}", size
        finally:
            os.remove(filename)


class Project:
    """
    API to manage a collection of key-report pairs persisted in a hub storage.

    It communicates with the Probabl's ``skore hub`` storage, based on the pickle
    representation. Its constructor initializes a hub project by creating a new
    project or by loading an existing one from a defined tenant.

    The class main methods are :func:`~skore_hub_project.Project.put`,
    :func:`~skore_hub_project.reports.metadata` and
    :func:`~skore_hub_project.reports.get`, respectively to insert a key-report pair
    into the Project, to obtain the reports metadata and to get a specific report.

    Parameters
    ----------
    tenant : str
        The tenant of the project.

        A tenant is a ``skore hub`` concept that must be configured on the
        ``skore hub`` interface. It represents an isolated entity managing users,
        projects, and resources. It can be a company, organization, or team that
        operates independently within the system.
    name : str
        The name of the project.

    Attributes
    ----------
    tenant : str
        The tenant of the project.
    name : str
        The name of the project.
    run_id : str
        The current run identifier of the project.
    """

    def __init__(self, tenant: str, name: str):
        """
        Initialize a hub project.

        Initialize a hub project by creating a new project or by loading an existing
        one from a defined tenant.

        Parameters
        ----------
        tenant : Path
            The tenant of the project.

            A tenant is a ``skore hub`` concept that must be configured on the
            ``skore hub`` interface. It represents an isolated entity managing users,
            projects, and resources. It can be a company, organization, or team that
            operates independently within the system.
        name : str
            The name of the project.
        """
        self.__tenant = tenant
        self.__name = name

    @property
    def tenant(self) -> str:
        """The tenant of the project."""
        return self.__tenant

    @property
    def name(self) -> str:
        """The name of the project."""
        return self.__name

    @cached_property
    def run_id(self) -> str:
        """The current run identifier of the project."""
        with AuthenticatedClient(raises=True) as client:
            request = client.post(f"projects/{self.tenant}/{self.name}/runs")
            run = request.json()

            return run["id"]

    def put(
        self,
        key: str,
        report: EstimatorReport,
        *,
        chunk_size: int = int(1e7),
        max_workers: int = 3,
        disable_progress_bar: bool = False,
    ):
        """
        Put a key-report pair to the hub project.

        If the key already exists, its last report is modified to point to this new
        report, while keeping track of the report history.

        Parameters
        ----------
        key : str
            The key to associate with ``report`` in the hub project.
        report : skore.EstimatorReport
            The report to associate with ``key`` in the hub project.
        chunk_size : int, optional
            The maximum size of chunks to upload in bytes, default ~10mb.
        max_workers : int, optional
            The maximum number of chunks uploaded in concurrence, default 3.
        disable_progress_bar : bool, optional
            Disable the progress bar that is displayed during upload, default False.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        from skore import EstimatorReport

        if not isinstance(report, EstimatorReport):
            raise TypeError(
                f"Report must be a `skore.EstimatorReport` (found '{type(report)}')"
            )

        # Pickle report in a tmpfile to avoid RAM overhead.
        with dumps(report) as (filename, checksum, size):
            # Ask for upload url if not already uploaded.
            with AuthenticatedClient(raises=True) as client:
                response = client.post(
                    url=f"projects/{self.tenant}/{self.name}/artefacts",
                    json=[
                        {
                            "checksum": checksum,
                            "content_type": "estimator-report-pickle",
                            "chunk_number": ceil(size / chunk_size),
                        }
                    ],
                )

            urls = response.json()

            if urls:
                etags = {}
                progress = Progress(
                    TextColumn("[bold cyan blink]Uploading..."),
                    BarColumn(
                        complete_style="dark_orange",
                        finished_style="dark_orange",
                        pulse_style="orange1",
                    ),
                    TextColumn("[orange1]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    transient=True,
                    disable=disable_progress_bar,
                )

                def put(client, url, filename, offset, size):
                    with open(filename, "rb") as file:
                        file.seek(offset)

                        response = client.put(
                            url=url,
                            content=file.read(size),
                            headers={"Content-Type": "application/octet-stream"},
                            timeout=None,
                        )

                        return response.headers["etag"]

                # Upload each chunk of the pickled report.
                with Client() as client, ThreadPoolExecutor(max_workers) as pool:
                    task_to_chunk_id = {}

                    for url in urls:
                        chunk_id = url["chunk_id"] or 1
                        task = pool.submit(
                            put,
                            client=client,
                            url=url["upload_url"],
                            filename=filename,
                            offset=((chunk_id - 1) * chunk_size),
                            size=chunk_size,
                        )

                        task_to_chunk_id[task] = chunk_id

                    try:
                        with progress:
                            etags = dict(
                                sorted(
                                    (task_to_chunk_id[task], task.result())
                                    for task in progress.track(
                                        as_completed(task_to_chunk_id),
                                        total=len(task_to_chunk_id),
                                    )
                                )
                            )
                    except BaseException:
                        for task in task_to_chunk_id:
                            task.cancel()

                        raise

                # Acknowledgement of sending.
                with AuthenticatedClient(raises=True) as client:
                    client.post(
                        url=f"projects/{self.tenant}/{self.name}/artefacts/complete",
                        json=[{"checksum": checksum, "etags": etags}],
                    )

        # Send metadata.
        with AuthenticatedClient(raises=True) as client:
            client.post(
                url=f"projects/{self.tenant}/{self.name}/items",
                json=dict(
                    (
                        *skore_estimator_report_item.Metadata(report),
                        (
                            "related_items",
                            list(skore_estimator_report_item.Representations(report)),
                        ),
                        ("parameters", {"checksum": checksum}),
                        ("key", key),
                        ("run_id", self.run_id),
                    )
                ),
            )

    @property
    def reports(self):
        """Accessor for interaction with the persisted reports."""

        def get(id: str) -> EstimatorReport:
            """Get a persisted report by its id."""
            # Retrieve report metadata.
            with AuthenticatedClient(raises=True) as client:
                response = client.get(
                    url=f"projects/{self.tenant}/{self.name}/experiments/estimator-reports/{id}"
                )

            metadata = response.json()
            checksum = metadata["raw"]["checksum"]

            # Ask for read url.
            with AuthenticatedClient(raises=True) as client:
                response = client.get(
                    url=f"projects/{self.tenant}/{self.name}/artefacts/read",
                    params={"artefact_checksum": [checksum]},
                )

            url = response.json()[0]["url"]

            # Download pickled report before unpickling it.
            #
            # It uses streaming responses that do not load the entire response body into
            # memory at once.
            with (
                TemporaryFile(mode="w+b") as tmpfile,
                Client() as client,
                client.stream(method="GET", url=url, timeout=None) as response,
            ):
                for data in response.iter_bytes():
                    tmpfile.write(data)

                tmpfile.seek(0)

                return joblib.load(tmpfile)

        def metadata() -> list[Metadata]:
            """Obtain metadata for all persisted reports regardless of their run."""

            def dto(summary):
                metrics = {
                    metric["name"]: metric["value"]
                    for metric in summary["metrics"]
                    if metric["data_source"] in (None, "test")
                }

                return {
                    "id": summary["id"],
                    "run_id": summary["run_id"],
                    "key": summary["key"],
                    "date": summary["created_at"],
                    "learner": summary["estimator_class_name"],
                    "dataset": summary["dataset_fingerprint"],
                    "ml_task": summary["ml_task"],
                    "rmse": metrics.get("rmse"),
                    "log_loss": metrics.get("log_loss"),
                    "roc_auc": metrics.get("roc_auc"),
                    "fit_time": metrics.get("fit_time"),
                    "predict_time": metrics.get("predict_time"),
                }

            with AuthenticatedClient(raises=True) as client:
                response = client.get(
                    f"projects/{self.tenant}/{self.name}/experiments/estimator-reports"
                )

            return sorted(map(dto, response.json()), key=itemgetter("date"))

        # Ensure project is created by calling `self.run_id`
        return self.run_id and SimpleNamespace(get=get, metadata=metadata)

    def __repr__(self) -> str:  # noqa: D105
        return f"Project(mode='hub', name='{self.name}', tenant='{self.tenant}')"

    @staticmethod
    def delete(tenant: str, name: str):
        """
        Delete a hub project.

        Parameters
        ----------
        tenant : Path
            The tenant of the project.

            A tenant is a ``skore hub`` concept that must be configured on the
            ``skore hub`` interface. It represents an isolated entity managing users,
            projects, and resources. It can be a company, organization, or team that
            operates independently within the system.
        name : str
            The name of the project.
        """
        with AuthenticatedClient(raises=True) as client:
            try:
                client.delete(f"projects/{tenant}/{name}")
            except HTTPStatusError as e:
                if e.response.status_code == 403:
                    raise PermissionError(
                        f"Failed to delete the project; "
                        f"please contact the '{tenant}' owner"
                    ) from e
                raise
