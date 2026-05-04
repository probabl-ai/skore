import functools
import shutil
import subprocess


def git_available() -> bool:
    """Check whether Git is installed and usable.

    Returns
    -------
    bool
        True if the ``git`` executable is found on the system path.
    """
    return shutil.which("git") is not None


def working_tree_clean() -> bool:
    """Check whether the Git working tree is clean.

    Assumes that ``git`` is available on the system path.

    Returns
    -------
    bool
        True if there are no uncommitted changes, False otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return False
    return result.stdout.strip() == ""


def head_commit_hash() -> str | None:
    """Obtain the hash of the HEAD commit.

    Assumes that ``git`` is available on the system path.

    Returns
    -------
    str or None
        The full SHA-1 hash of the current HEAD commit, if available.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip()


@functools.cache
def git_commit() -> str | None:
    """Obtain the hash of the latest Git commit.

    Returns
    -------
    str or None
        The commit hash, if available.

    Examples
    --------
    >>> # xdoctest: +SKIP

    With git not available:

    >>> git_commit()
    None

    With a clean working tree:

    >>> git_commit()
    e5de78623d1227d8b7f90f949bb7830eb9433ade

    With a dirty working tree (uncommitted changes):

    >>> git_commit()
    e5de78623d1227d8b7f90f949bb7830eb9433ade (working tree dirty)
    """
    if not git_available():
        return None
    commit_hash = head_commit_hash()
    if working_tree_clean():
        return commit_hash
    return f"{commit_hash} (working tree dirty)"
