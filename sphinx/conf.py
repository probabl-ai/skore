# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
from sphinx_gallery.sorting import ExplicitOrder

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("sphinxext"))
from github_link import make_linkcode_resolve  # noqa
from matplotlib_skore_scraper import matplotlib_skore_scraper  # noqa

project = "skore"
copyright = "2024, Probabl"
author = "Probabl"
version = os.environ["SPHINX_VERSION"]
release = os.environ["SPHINX_RELEASE"]
domain = os.environ["SPHINX_DOMAIN"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinx_autosummary_accessors",
]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# Add any paths that contain templates here, relative to this directory.
autosummary_generate = True  # generate stubs for all classes
templates_path = ["_templates"]

autodoc_typehints = "none"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]
html_js_files = [
    "js/sg_plotly_resize.js",
]

# list of examples in explicit order
subsections_order = [
    "../examples/getting_started",
    "../examples/use_cases",
    "../examples/model_evaluation",
    "../examples/skore_project",
    "../examples/technical_details",
]

# sphinx_gallery options
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to example scripts
    "gallery_dirs": "auto_examples",  # path to gallery generated output
    "filename_pattern": "plot_",  # pattern to select examples; change this to only build some of the examples
    "subsection_order": ExplicitOrder(subsections_order),  # sorting gallery subsections
    # see https://sphinx-gallery.github.io/stable/configuration.html#sub-gallery-order
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
    # "image_scrapers": ['matplotlib'], # use in-built scraper
    "image_scrapers": [matplotlib_skore_scraper()], # alternative option using custom class
    "abort_on_example_error": True,
}

# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "skrub": ("https://skrub-data.org/stable/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable", None),
    "polars": ("https://docs.pola.rs/py-polars/html", None),
    "seaborn": ("http://seaborn.pydata.org", None),
}

numpydoc_show_class_members = False

html_title = "skore"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
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
        "json_url": f"https://{domain}/versions.json",
        "version_match": release,
    },
    "check_switcher": True,
    "show_version_warning_banner": True,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

# Plausible Analytics
html_theme_options["analytics"] = {
    # The domain you'd like to use for this analytics instance
    "plausible_analytics_domain": domain,
    # The analytics script that is served by Plausible
    "plausible_analytics_url": "https://plausible.io/js/script.js",
}

# Sphinx remove the sidebar from some pages
html_sidebars = {
    "index": [],
    "install": [],
    "contributing": [],
}

# Sphinx-Copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Options for github link for what's new -----------------------------------

# Config for sphinx_issues
issues_uri = "https://github.com/probabl-ai/skore/issues/{issue}"
issues_github_path = "probabl-ai/skore"
issues_user_uri = "https://github.com/{user}"

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "skore",
    (
        "https://github.com/probabl-ai/"
        "skore/blob/{revision}/"
        "{package}/src/skore/{path}#L{lineno}"
    ),
)
