"""Cross-check stable contracts between the public API and documentation."""

import re
from pathlib import Path

import skore
from skore._checks.builtin import _BUILTIN_CHECKS

SPHINX_DIR = Path(__file__).parents[3] / "sphinx"


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
