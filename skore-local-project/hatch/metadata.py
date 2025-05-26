"""Hatchling hooks used to dynamically update the metadata of the package."""

from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface


def readlines(filepath):
    """Yied the content of ``filepath`` line by line."""
    with open(filepath) as file:
        yield from file


class MetadataHook(MetadataHookInterface):
    """Hatchling hooks used to dynamically update the metadata of the package."""

    def update(self, metadata):
        """
        Update the metadata of the package, after it has been loaded.

        Update ``LICENSE`` and ``README`` from root files, shared in the GH repository.
        Update ``VERSION`` from environment variables filled by the GH release pipeline.

        Notes
        -----
        Example of configuration:

            {
                'path': 'hatch/metadata.py',
                'version-default': '0.0.0+unknown',
                'license': {'file': '../LICENSE'},
                'readme': {'file': '../README.md'},
            }
        """
        # Retrieve LICENCE from root files
        license_filepath = self.config["license"]["file"]
        license = Path(self.root, license_filepath).read_text(encoding="utf-8")

        # Copy LICENCE file in `sdist`
        with open(Path(self.root, "LICENSE"), "w") as f:
            f.write(license)

        # Retrieve README from root files
        readme_filepath = self.config["readme"]["file"]
        readme = Path(self.root, readme_filepath).read_text(encoding="utf-8")

        # Retrieve VERSION from file created by the CI
        try:
            version = Path(self.root, "VERSION.txt").read_text(encoding="utf-8")
        except FileNotFoundError:
            version = self.config["version-default"]

        # Update metadata
        metadata["license"] = {"text": license, "content-type": "text/plain"}
        metadata["readme"] = {"text": readme, "content-type": "text/markdown"}
        metadata["version"] = version
