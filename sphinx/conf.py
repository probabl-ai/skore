# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
from sphinx_gallery.sorting import ExplicitOrder

project = "skore"
copyright = "2024, Probabl"
author = "Probabl"
version = os.environ["SPHINX_VERSION"]
release = os.environ["SPHINX_RELEASE"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
]
templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

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
examples_ordered = [
    "../examples/getting_started",
    "../examples/getting_started/plot_quick_start",
    "../examples/getting_started/plot_skore_product_tour",
    "../examples/getting_started/plot_working_with_projects",
    "../examples/getting_started/plot_tracking_items",
    "../examples/model_evaluation",
    "../examples/model_evaluation/plot_train_test_split"
    "../examples/model_evaluation/plot_cross_validate",
]

# sphinx_gallery options
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to example scripts
    "gallery_dirs": "auto_examples",  # path to gallery generated output
    "subsection_order": ExplicitOrder(examples_ordered),
    "within_subsection_order": "FileNameSortKey",  # See https://sphinx-gallery.github.io/stable/configuration.html#sorting-gallery-examples for alternatives
    "show_memory": False,
    "write_computation_times": False,
    "reference_url": {
        # The module you locally document uses None
        "skore": None,
    },
    "backreferences_dir": "generated",
    "doc_module": "skore",
    "default_thumb_file": "./_static/images/Logo_Skore_Light@2x.svg",
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
        "index": [], # hide secondary sidebar items for the landing page
        "install": [],
    },
    "external_links": [
        {
            "url": "https://probabl.ai",
            "name": "Probabl",
        },
    ],
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/probabl-ai/skore/",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/scBZerAGwW",
            "icon": "fa-brands fa-discord",
        },
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/company/probabl/",
            "icon": "fa-brands fa-linkedin-in",
        },
        {
            "name": "Bluesky",
            "url": "https://bsky.app/profile/probabl.ai",
            "icon": "fa-brands fa-bluesky",
        },
        {
            "name": "X (ex-Twitter)",
            "url": "https://x.com/probabl_ai",
            "icon": "fa-brands fa-x-twitter",
        },
    ],
    "switcher": {
        "json_url": "https://skore.probabl.ai/versions.json",
        "version_match": version,
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
    "plausible_analytics_domain": "skore.probabl.ai",
    # The analytics script that is served by Plausible
    "plausible_analytics_url": "https://plausible.io/js/script.js",
}

# Sphinx remove the sidebar from some pages
html_sidebars = {
    "index": [],
    "install": [],
    "contributing_link": [],
}

# Sphinx-Copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
