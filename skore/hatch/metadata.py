import shutil
from pathlib import Path
from contextlib import ExitStack, suppress

from hatchling.metadata.plugin.interface import MetadataHookInterface


# @dataclasses.dataclass
# class Configuration:


def readlines(filepath):
    with open(filepath) as file:
        yield from file


class MetadataHook(MetadataHookInterface):
    def update(self, metadata):
        license_path = Path(self.root, self.config["license-file"])
        license = license_path.read_text(encoding="utf-8")
        readme = Path(self.root, self.config["readme-file"]).read_text(encoding="utf-8")
        version = self.config["version-default"]

        with suppress(FileNotFoundError):
            version = Path(self.root, "VERSION.txt").read_text(encoding="utf-8")

        metadata["license"] = {"text": license, "content-type": "text/plain"}
        metadata["readme"] = {"text": readme, "content-type": "text/markdown"}
        metadata["version"] = version

        license_dest = Path(self.root, license_path.name)
        shutil.copy2(license_path, license_dest)

        # {
        #     'path': 'hatch/metadata.py',
        #     'license-file': '../LICENSE',
        #     'readme-file': '../README.md',
        #     'version-default': '0.0.0+unknown',
        #     'dependencies': {'file': 'requirements.in'},
        #     'optional-dependencies': {
        #         'test': {'file': 'test-requirements.in'},
        #         'sphinx': {'file': 'sphinx-requirements.in'}
        #     }
        # }

        metadata["dependencies"] = list(
            map(str.strip, readlines(self.config["dependencies"]))
        )
        metadata["optional-dependencies"] = {
            label: list(map(str.strip, readlines(filepath)))
            for label, filepath in self.config["optional-dependencies"].items()
        }
