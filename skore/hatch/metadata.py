from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface


def readlines(filepath):
    with open(filepath) as file:
        yield from file


class MetadataHook(MetadataHookInterface):
    def update(self, metadata):
        """
        Notes
        -----
        Example of configuration:

            {
                'path': 'hatch/metadata.py',
                'version-default': '0.0.0+unknown',
                'license': {'file': '../LICENSE'},
                'readme': {'file': '../README.md'},
                'dependencies': {'file': 'requirements.in'},
                'optional-dependencies': {
                    'test': {'file': 'test-requirements.in'},
                    'sphinx': {'file': 'sphinx-requirements.in'},
                }
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

        # Retrieve dependencies from requirements.in
        dependencies_filepath = self.config["dependencies"]["file"]
        dependencies = list(map(str.strip, readlines(dependencies_filepath)))

        # Retrieve optional dependencies from *-requirements.in
        optional_dependencies_label_to_filepath = self.config["optional-dependencies"]
        optional_dependencies = {}

        for label, filepath in optional_dependencies_label_to_filepath.items():
            optional_dependencies_filepath = filepath["file"]
            optional_dependencies[label] = list(
                map(
                    str.strip,
                    readlines(optional_dependencies_filepath),
                )
            )

        # Update metadata
        metadata["license"] = {"text": license, "content-type": "text/plain"}
        metadata["readme"] = {"text": readme, "content-type": "text/markdown"}
        metadata["version"] = version
        metadata["dependencies"] = dependencies
        metadata["optional-dependencies"] = optional_dependencies
