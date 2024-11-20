# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "skore"
copyright = "2024, Probabl"
author = "Probabl"
# version = "0"
# release = "0"

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

class SubSectionTitleOrder:
    """Sort example gallery by title of subsection.

    Assumes README.txt exists for all subsections and uses the subsection with
    dashes, '---', as the adornment.
    """

    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.regex = re.compile(r"^([\w ]+)\n-", re.MULTILINE)

    def __repr__(self):
        return "<%s>" % (self.__class__.__name__,)

    def __call__(self, directory):
        src_path = os.path.normpath(os.path.join(self.src_dir, directory))

        # Forces Release Highlights to the top
        if os.path.basename(src_path) == "release_highlights":
            return "0"

        readme = os.path.join(src_path, "README.txt")

        try:
            with open(readme, "r") as f:
                content = f.read()
        except FileNotFoundError:
            return directory

        title_match = self.regex.search(content)
        if title_match is not None:
            return title_match.group(1)
        return directory

sg_examples_dir = "../examples"


# sphinx_gallery options
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to example scripts
    "gallery_dirs": "auto_examples",  # path to gallery generated output
    "within_subsection_order": "FileNameSortKey",  # See https://sphinx-gallery.github.io/stable/configuration.html#sorting-gallery-examples for alternatives
    "show_memory": False,
    "write_computation_times": False,
    'reference_url': {
        # The module you locally document uses None
        'skore': None,
        },
    "subsection_order": SubSectionTitleOrder(sg_examples_dir),
}

# intersphinx configuration
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

numpydoc_show_class_members = False

html_title = "skore"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # "logo": {
    #     "image_relative": "_static/skrub.svg",
    #     "image_light": "_static/skrub.svg",
    #     "image_dark": "_static/skrub.svg",
    # },
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