"""Process primitives shared by harness launchers."""

from __future__ import annotations

import os
import shutil
import subprocess  # noqa: F401  (tests patch this module attribute)
import sys  # noqa: F401  (tests patch this module attribute)
from pathlib import Path  # noqa: F401  (tests patch this module attribute)


def _exec_harness(
    name: str, argv: list[str], *, env: dict[str, str] | None = None
) -> None:
    executable = shutil.which(argv[0])
    if executable is None:
        raise RuntimeError(f"{name} is not installed or not on PATH.")
    os.execve(executable, argv, env or os.environ)
