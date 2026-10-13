"""Cross-check stable contracts between the public API and documentation."""

import re
import tomllib
from pathlib import Path

import skore
from skore._checks.builtin import _BUILTIN_CHECKS

REPO_DIR = Path(__file__).parents[3]
SPHINX_DIR = REPO_DIR / "sphinx"


def test_public_api_is_mentioned_in_reference():
    """Every top-level public symbol is discoverable from the reference."""
    reference_dir = SPHINX_DIR / "reference"
    reference_text = "\n".join(
        path.read_text()
        for path in reference_dir.rglob("*.rst")
        if "api" not in path.relative_to(reference_dir).parts
    )

    missing = [
        name
        for name in skore.__all__
        if re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", reference_text) is None
    ]

    assert missing == []


def test_builtin_check_documentation_anchors_exist():
    """Every built-in check links to an anchor in the automated-checks guide."""
    guide = (SPHINX_DIR / "user_guide" / "automated_checks.rst").read_text()

    missing = {
        check.code: check.docs_url
        for check in _BUILTIN_CHECKS
        if check.docs_url is not None
        and not check.docs_url.startswith("http")
        and f".. _{check.docs_url}:" not in guide
    }

    assert missing == {}


def test_builtin_check_titles_match_user_guide():
    """User-guide headings use the title emitted by each built-in check."""
    guide = (SPHINX_DIR / "user_guide" / "automated_checks.rst").read_text()
    missing = [
        f"{check.code} - {check.title}"
        for check in _BUILTIN_CHECKS
        if f"{check.code} - {check.title}" not in guide
    ]

    assert missing == []


def test_documented_python_requirement_matches_package_metadata():
    """Install instructions use the minimum version declared by the package."""
    pyproject = tomllib.loads((REPO_DIR / "skore" / "pyproject.toml").read_text())
    match = re.search(r">=([0-9.]+)", pyproject["project"]["requires-python"])
    assert match is not None
    minimum = match.group(1)

    documented = {
        "README.md": (REPO_DIR / "README.md").read_text(),
        "CONTRIBUTING.rst": (REPO_DIR / "CONTRIBUTING.rst").read_text(),
        "sphinx/install.rst": (SPHINX_DIR / "install.rst").read_text(),
    }
    missing = [path for path, text in documented.items() if minimum not in text]

    assert missing == []
