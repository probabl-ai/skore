from platform import python_version
from typing import Literal

from skore._plugins.hub.artifact.artifact import Artifact
from skore._plugins.requirements import infer


class Environment(Artifact):
    content_type: Literal["text/plain"] = "text/plain"
    python_version: str = python_version()

    def content_to_upload(self) -> str | None:
        try:
            return "\n".join(
                f"{requirement['name']}=={requirement['version']}"
                for requirement in infer()
            )
        except Exception:
            return None
