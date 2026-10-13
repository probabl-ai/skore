"""Project-local ``.skore`` configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

SKORE_FILENAME = ".skore"


class SkoreConfigError(ValueError):
    """A ``.skore`` file is present but cannot be loaded."""


@dataclass
class SkoreConfig:
    """Persisted Skore agent settings for a project."""

    hub_url: str
    workspace: str
    workspace_id: int
    harness: str | None = None

    @classmethod
    def load(cls, path: Path) -> SkoreConfig | None:
        """Load ``.skore`` from ``path`` when present and valid.

        Parsing does not change ``SKORE_HUB_URI``. Callers that want the saved
        hub for this process apply it explicitly.
        """
        file_path = path / SKORE_FILENAME
        if not file_path.is_file():
            return None
        try:
            data = json.loads(file_path.read_text() or "{}")
        except json.JSONDecodeError as error:
            raise SkoreConfigError(f"could not parse {file_path}: {error}") from error
        if not isinstance(data, dict):
            raise SkoreConfigError(f"{file_path} does not hold a JSON object.")
        hub_url = data.get("hub_url")
        workspace = data.get("workspace")
        workspace_id = data.get("workspace_id")
        if not hub_url or not workspace or workspace_id is None:
            raise SkoreConfigError(
                f"{file_path} is missing hub_url, workspace, or workspace_id."
            )
        try:
            parsed_id = int(workspace_id)
        except (TypeError, ValueError) as error:
            raise SkoreConfigError(
                f"{file_path} workspace_id is not an integer."
            ) from error
        from skore._cli.agents import normalize_harness_name

        harness = normalize_harness_name(data.get("harness"))
        return cls(
            hub_url=hub_url,
            workspace=workspace,
            workspace_id=parsed_id,
            harness=harness,
        )

    def save(self, path: Path) -> Path:
        """Write this config to ``path/.skore`` and return the file path."""
        file_path = path / SKORE_FILENAME
        payload = {
            key: value for key, value in asdict(self).items() if value is not None
        }
        file_path.write_text(json.dumps(payload, indent=2) + "\n")
        return file_path


def ensure_gitignore_entry(workspace: Path, entry: str = SKORE_FILENAME) -> None:
    """Append ``entry`` to the project ``.gitignore`` when missing."""
    gitignore = workspace / ".gitignore"
    if gitignore.is_file():
        lines = gitignore.read_text().splitlines()
        if any(line.strip() == entry for line in lines):
            return
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(entry)
        gitignore.write_text("\n".join(lines) + "\n")
        return
    gitignore.write_text(f"{entry}\n")
