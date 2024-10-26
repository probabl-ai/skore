# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "skore"
copyright = "2024, Probabl"
author = "Probabl"
version = "0"
release = "0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
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
html_js_files = []

# sphinx_gallery options
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to example scripts
    "gallery_dirs": "auto_examples",  # path to gallery generated output
    "within_subsection_order": "FileNameSortKey",  # See https://sphinx-gallery.github.io/stable/configuration.html#sorting-gallery-examples for alternatives
    "show_memory": False,
    "write_computation_times": False,
}

# intersphinx configuration
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

numpydoc_show_class_members = False

html_title = "skore"

# From skrub
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # "logo": {
    #     "image_relative": "_static/skrub.svg",
    #     "image_light": "_static/skrub.svg",
    #     "image_dark": "_static/skrub.svg",
    # },
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
}
