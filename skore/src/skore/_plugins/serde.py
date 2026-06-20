import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

import joblib

ArtifactId: TypeAlias = str


@dataclass
class JoblibPicklePointer:
    """A lazy reference to an artifact serialized with joblib.

    A pointer replaces a heavy payload inside a report state so the bytes can
    live elsewhere (typically content-addressed storage that de-duplicates
    identical payloads across reports).
    """

    artifact_id: ArtifactId


def dumps(artifact: Any) -> bytes:
    """Serialize the artifact into bytes suitable for storage."""
    with io.BytesIO() as stream:
        joblib.dump(artifact, stream)
        return stream.getvalue()


def loads(artifact_bytes: bytes) -> Any:
    """Deserialize the artifact from its stored byte representation."""
    with io.BytesIO(artifact_bytes) as stream:
        return joblib.load(stream)


def externalize(
    state: dict[str, Any],
    artifact_writer: Callable[[ArtifactId, bytes], None],
) -> dict[str, Any]:
    """Extract large artifacts in ``state`` to external storage.

    Replace large artifacts in ``state`` with pointers.

    Parameters
    ----------
    state : dict[str, Any]
        Report state.

    artifact_writer : callable
        Called as ``artifact_writer(id, bytes)`` for each artifact extracted
        from ``state``; responsible for persisting ``bytes`` under ``id``.
    """
    new_data: dict[str, JoblibPicklePointer] = {}
    for key, obj in state["data"].items():
        obj_bytes = dumps(obj)
        artifact_id = joblib.hash(obj_bytes)
        artifact_writer(artifact_id, obj_bytes)
        new_data[key] = JoblibPicklePointer(artifact_id)
    return state | {"data": new_data}


def internalize(
    state: Any,
    artifact_reader: Callable[[ArtifactId], bytes],
) -> Any:
    """Replace artifact pointers in ``state`` with the data they point to.

    Parameters
    ----------
    state : Any
        Possibly-nested structure (dict, list, or scalar) that may contain
        :class:`JoblibPicklePointer` instances; traversed recursively.

    artifact_loader : callable
        Called as ``artifact_loader(artifact_id)`` for each pointer encountered.
    """
    if isinstance(state, JoblibPicklePointer):
        artifact_bytes = artifact_reader(state.artifact_id)
        return loads(artifact_bytes)
    elif isinstance(state, dict):
        return {k: internalize(v, artifact_reader) for k, v in state.items()}
    elif isinstance(state, list):
        return [internalize(v, artifact_reader) for v in state]
    else:
        return state
