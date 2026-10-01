"""Use a local Python inventory when docs.python.org cannot be reached."""

from __future__ import annotations

import io
import re
import tarfile
import urllib.request
from pathlib import Path

DOCS_INVENTORY = "https://docs.python.org/3/objects.inv"
FTP_INDEX = "https://www.python.org/ftp/python/doc/"
DEST = Path(__file__).parent / "_intersphinx" / "python-objects.inv"
_VERSION = re.compile(r'href="(3\.\d+\.\d+)/"')


def _reachable(url: str) -> bool:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status == 200
    except Exception:
        return False


def _latest_html_tarball() -> str:
    with urllib.request.urlopen(FTP_INDEX, timeout=30) as response:
        listing = response.read().decode()
    versions = sorted(
        {tuple(int(part) for part in match.split(".")) for match in _VERSION.findall(listing)},
        reverse=True,
    )
    for version in versions:
        version_s = ".".join(str(part) for part in version)
        url = f"{FTP_INDEX}{version_s}/python-{version_s}-docs-html.tar.bz2"
        if _reachable(url):
            return url
    raise SystemExit("Could not find a Python HTML documentation tarball.")


def _download_inventory() -> None:
    url = _latest_html_tarball()
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:bz2") as archive:
        member = next(item for item in archive.getmembers() if item.name.endswith("/objects.inv"))
        extracted = archive.extractfile(member)
        if extracted is None:
            raise SystemExit(f"Inventory missing from {url}.")
        contents = extracted.read()
    DEST.parent.mkdir(exist_ok=True)
    DEST.write_bytes(contents)


def main() -> None:
    if _reachable(DOCS_INVENTORY):
        DEST.unlink(missing_ok=True)
        return
    if DEST.is_file():
        return
    _download_inventory()


if __name__ == "__main__":
    main()
