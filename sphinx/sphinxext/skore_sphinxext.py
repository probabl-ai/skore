from __future__ import annotations

from typing import Any

from sphinx.application import Sphinx

from .github_link import make_linkcode_resolve
from .matplotlib_skore_scraper import matplotlib_skore_scraper


def _config_inited(app: Sphinx, config: Any) -> None:
    config.linkcode_resolve = make_linkcode_resolve(
        "skore",
        (
            "https://github.com/probabl-ai/"
            "skore/blob/{revision}/"
            "{package}/src/skore/{path}#L{lineno}"
        ),
    )

    if getattr(config, "sphinx_gallery_conf", None) is not None:
        config.sphinx_gallery_conf["image_scrapers"] = [matplotlib_skore_scraper()]


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("config-inited", _config_inited)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
