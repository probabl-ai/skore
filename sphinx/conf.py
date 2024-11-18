# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "skore"
copyright = "2024, Probabl"
author = "Probabl"
version = "0.3"  # dynamic
release = "0.3.0"  # dynamic

# 0.1/
# 0.2/
# 0.3/
# dev/
# index.html
# versions.json

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

# sphinx_gallery options
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to example scripts
    "gallery_dirs": "auto_examples",  # path to gallery generated output
    "within_subsection_order": "FileNameSortKey",  # See https://sphinx-gallery.github.io/stable/configuration.html#sorting-gallery-examples for alternatives
    "show_memory": False,
    "write_computation_times": False,
    "reference_url": {
        # The module you locally document uses None
        "skore": None,
    },
}

# intersphinx configuration
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

numpydoc_show_class_members = False

html_title = "skore"

# html_additional_pages = {"index": "index.html"}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # When specified as a dictionary, the keys should follow glob-style patterns, as in
    # https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-exclude_patterns
    # In particular, "**" specifies the default for all pages
    # Use :html_theme.sidebar_secondary.remove: for file-wide removal
    "secondary_sidebar_items": {
        "**": ["page-toc", "sourcelink", "sg_download_links", "sg_launcher_links"]
    },
    "external_links": [
        {
            "url": "https://probabl.ai",
            "name": "Probabl website",
        },
    ],
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "Twitter",
            "url": "https://x.com/probabl_ai",
            "icon": "fa-brands fa-twitter",
        },
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
    ],
    "announcement": "This code is still in development. <strong>The API is subject to change.</strong>",
    "switcher": {"json_url": "_static/switcher.json", "version_match": version},
    "check_switcher": True,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher"],
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
    "install": [],
    "getting_started": [],
    "user_guide": [],
    "contributor_guide": [],
}

# Sphinx-Copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
