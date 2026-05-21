import io
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

import joblib


@dataclass
class ArtifactPointer:
    artifact_id: str

    @staticmethod
    def load(artifact_bytes: bytes) -> Any:
        raise NotImplementedError()

    @staticmethod
    def dump(artifact: Any) -> bytes:
        raise NotImplementedError()


class JoblibPointer(ArtifactPointer):
    @staticmethod
    def load(artifact_bytes: bytes) -> Any:
        with io.BytesIO(artifact_bytes) as stream:
            return joblib.load(stream)

    @staticmethod
    def dump(artifact: Any) -> bytes:
        with io.BytesIO() as stream:
            joblib.dump(artifact, stream)
            return stream.getvalue()


def split_data_from_state(
    state: dict[str, Any],
) -> Generator[tuple[str, bytes], None, None]:
    """Extract data artifacts from a report state.

    The input ``state`` is updated in place: each entry in ``state["data"]`` is
    replaced by an artifact pointer, and the serialized artifact bytes are yielded.
    """
    data = {}
    for key, obj in state["data"].items():
        obj_bytes = JoblibPointer.dump(obj)
        bytes_id = joblib.hash(obj_bytes)
        yield (bytes_id, obj_bytes)
        data[key] = JoblibPointer(bytes_id)
    state |= {"data": data}


def restore_data_in_state(
    state: Any,
    artifact_loader: Callable[[str], bytes],
) -> Any:
    """Restore artifact pointers in a report state.

    Pointers are loaded with ``artifact_loader`` and deserialized; dictionaries
    and lists are traversed recursively while other values are returned unchanged.
    """
    if isinstance(state, ArtifactPointer):
        artifact_bytes = artifact_loader(state.artifact_id)
        return state.load(artifact_bytes)
    elif isinstance(state, dict):
        return {k: restore_data_in_state(v, artifact_loader) for k, v in state.items()}
    elif isinstance(state, list):
        return [restore_data_in_state(v, artifact_loader) for v in state]
    else:
        return state
