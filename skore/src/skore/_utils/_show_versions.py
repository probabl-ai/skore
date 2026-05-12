"""
Utility methods to print system info for debugging.

adapted from :func:`sklearn.show_versions`
"""

import platform
import re
import sys
from collections import defaultdict
from typing import Any

# PEP 508 markers: extract `extra == "…"` for optional dependency groups.
_EXTRA_MARKER_RE = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")

# Optional dependency sections printed by :func:`show_versions` (plus Core).
_SHOW_VERSION_EXTRAS = frozenset({"hub", "jupyter", "local", "mlflow"})


def _extra_label_from_markers(markers: str) -> str:
    """Return optional dependency group name(s) from marker string, or ``-``."""
    if not markers.strip():
        return "-"
    found = _EXTRA_MARKER_RE.findall(markers)
    if not found:
        return "-"
    return ",".join(sorted(set(found)))


def _section_title(extra_label: str) -> str:
    """Human-readable heading for a dependency group."""
    return "skore" if extra_label == "-" else f"skore[{extra_label}]"


def _sort_extra_keys(extra_label: str) -> tuple[int, str]:
    """Core (no extra) first, then alphabetical extras."""
    return (0 if extra_label == "-" else 1, extra_label)


def _show_extra_section(extra_label: str) -> bool:
    """Whether this dependency group is included in :func:`show_versions` output."""
    if extra_label == "-":
        return True
    parts = {p.strip() for p in extra_label.split(",")}
    return bool(parts & _SHOW_VERSION_EXTRAS)


def _get_sys_info() -> dict[str, Any]:
    """System information.

    Returns
    -------
    sys_info : dict
        system and Python version information

    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info() -> list[tuple[str, str, Any]]:
    """Overview of the installed version of main dependencies.

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns
    -------
    deps_info: list of (name, extra, version)
        Package name, optional-extra label from metadata (``-`` if not from an
        extra), and installed version (or ``None`` if not installed).

    """
    from importlib.metadata import PackageNotFoundError, requires, version

    dependencies = ["pip", "skore", *(requires("skore") or [])]
    rows: list[tuple[str, str, Any]] = []

    for dependency in dependencies:
        head, _, markers = dependency.partition(";")
        head = head.strip()
        markers = markers.strip()
        name = re.split(r"\s*(?:<|>|=|!|~|\[)", head, maxsplit=1)[0].strip()
        extra_label = _extra_label_from_markers(markers)

        try:
            ver = version(name)
        except PackageNotFoundError:
            ver = None

        rows.append((name, extra_label, ver))

    return rows


def show_versions() -> None:
    """Print useful debugging information.

    Examples
    --------
    >>> # xdoctest: +SKIP
    >>> from skore import show_versions
    >>> show_versions()
    """
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")  # noqa: T201
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")  # noqa: T201

    print("\nPython dependencies:\n")  # noqa: T201
    by_extra: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    for name, extra, stat in deps_info:
        if name not in by_extra[extra]:
            by_extra[extra][name] = stat

    first_section = True
    for extra_label in sorted(by_extra, key=_sort_extra_keys):
        if not _show_extra_section(extra_label):
            continue
        if not first_section:
            print()  # noqa: T201
        first_section = False

        title = _section_title(extra_label)
        print(title)  # noqa: T201
        print("-" * len(title))  # noqa: T201
        for name, stat in by_extra[extra_label].items():
            print(f"{name:>13}: {stat}")  # noqa: T201
