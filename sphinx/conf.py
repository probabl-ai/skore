"""
Configuration file for the Sphinx documentation builder.

This file configures the Sphinx documentation build for skore, including
extensions, themes, and gallery settings.
"""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import pathlib
import sys

# Make it possible to load custom extensions from sphinxext directory
sys.path.append(str(pathlib.Path("sphinxext").resolve()))

project = "skore"
copyright = "2026, Probabl"
author = "Probabl"
version = os.environ["SPHINX_VERSION"]
release = os.environ["SPHINX_RELEASE"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_autosummary_accessors",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "sphinx_tabs.tabs",
    # Custom extensions
    "generate_accessor_tables",
    "github_link",
    "landing_page",
    "matplotlib_skore_scraper",
]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

# Configuration for generate_accessor_tables extension
accessor_summary_classes = [
    "skore.EstimatorReport",
    "skore.CrossValidationReport",
    "skore.ComparisonReport",
]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# Produce `plot::` directives for examples that contain `import matplotlib` or
# `from matplotlib import`.
numpydoc_use_plots = True

# Options for the `::plot` directive:
# https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html
plot_formats = ["png"]
plot_include_source = True
plot_html_show_formats = False
plot_html_show_source_link = False
plot_rcparams = {
    "savefig.bbox": "tight",  # to not crop the figure when shown in the docs
}
plot_apply_rcparams = True  # if context option is used

# Add any paths that contain templates here, relative to this directory.
autosummary_generate = True  # generate stubs for all classes
templates_path = ["_templates"]

autodoc_typehints = "none"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]
html_js_files = [
    "js/sg_plotly_resize.js"
]

# sphinx_gallery options
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to example scripts
    "gallery_dirs": "auto_examples",  # path to gallery generated output
    "filename_pattern": "plot_",
    "subsection_order": [
        "../examples/getting_started",
        "../examples/use_cases",
        "../examples/model_evaluation",
        "../examples/technical_details",
    ],
    "within_subsection_order": "ExampleTitleSortKey",  # See https://sphinx-gallery.github.io/stable/configuration.html#sorting-gallery-examples for alternatives
    "show_memory": False,
    "write_computation_times": False,
    "reference_url": {
        # The module you locally document uses None
        "skore": None,
    },
    "backreferences_dir": "reference/api",
    "doc_module": "skore",
    # "reset_modules": (reset_mpl, "seaborn"),
    "abort_on_example_error": True,
    "parallel": True,
}

# Build the HUB examples conditionally, only when credentials are available:
# - after each __commit__ on the `main` branch,
# - after each __release__ (tag `skore/X.Y.Z`).
if not (
    os.environ.get("GITHUB_ACTIONS")
    and os.environ.get("SPHINX_EXAMPLE_API_KEY")
    and os.environ.get("SPHINX_EXAMPLE_WORKSPACE")
):
    sphinx_gallery_conf["ignore_pattern"] = (
        r"plot_getting_started\.py|plot_skore_hub_project\.py"
    )

# Expose HUB URLs to RST
example_base_url = (os.environ.get("SPHINX_EXAMPLE_BASE_URL") or "https://example.com")
rst_epilog = f"""
.. _example-getting-started: {example_base_url}/example-getting-started-{version}/cross-validations
.. _example-skore-hub-project: {example_base_url}/example-skore-hub-project-{version}
"""

# Skore Hub base URL for landing page link (versioned example on Hub)
html_context = {
    "skore_hub_example_url": f"{example_base_url}/example-getting-started-{version}/cross-validations/",
}
# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "skrub": ("https://skrub-data.org/stable/", None),
    "pandas": ("http://pandas.pydata.org/docs/", None),
    "polars": ("https://docs.pola.rs/py-polars/html", None),
    "seaborn": ("http://seaborn.pydata.org", None),
    "xgboost": ("https://xgboost.readthedocs.io/en/stable/", None),
}

numpydoc_show_class_members = False

html_title = "skore"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "announcement": (
        "📣 Help shape the future of skore and the scikit-learn ecosystem by "
        '<a href="https://forms.gle/2fivh6RRrBF21CTD9" target="_blank" '
        'style="text-decoration: underline;">taking our survey</a>!'
    ),
    "logo": {
        "image_light": "_static/images/Logo_Skore_Light@2x.svg",
        "image_dark": "_static/images/Logo_Skore_Dark@2x.svg",
    },
    # When specified as a dictionary, the keys should follow glob-style patterns, as in
    # https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-exclude_patterns
    # In particular, "**" specifies the default for all pages
    # Use :html_theme.sidebar_secondary.remove: for file-wide removal
    "secondary_sidebar_items": {
        "**": ["page-toc", "sourcelink", "sg_download_links", "sg_launcher_links"],
        "index": [],  # hide secondary sidebar items for the landing page
        "install": [],
    },
    "external_links": [
        {
            "url": "https://probabl.ai",
            "name": "Probabl",
        },
    ],
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/probabl-ai/skore/",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discord",
            "url": "https://discord.probabl.ai",
            "icon": "fa-brands fa-discord",
        },
        {
            "name": "YouTube",
            "url": "https://youtube.com/playlist?list=PLSIzlWDI17bTpixfFkooxLpbz4DNQcam3",
            "icon": "fa-brands fa-youtube",
        },
    ],
    "switcher": {
        "json_url": f"{os.environ["SPHINX_URL"]}/versions.json",
        "version_match": release,
    },
    "check_switcher": True,
    "show_version_warning_banner": True,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {"index": "index.html"}

# Remove the sidebar from some pages
html_sidebars = {
    "index": [],
    "install": [],
    "contributing": [],
    "changelog": [],
}

# Sphinx-Copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# sphinx-issues config
issues_github_path = "probabl-ai/skore"
