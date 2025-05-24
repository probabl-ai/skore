# Minimal makefile for Sphinx documentation
#

# Ensures SPHINX_DOMAIN is not an empty string
ifeq ($(SPHINX_DOMAIN),)
	SPHINX_DOMAIN := docs.skore.probabl.ai
endif

export SPHINX_VERSION ?= dev
export SPHINX_RELEASE ?= 0.0.0+dev
export SPHINX_DOMAIN

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS  ?= --fail-on-warning
SPHINXBUILD ?= sphinx-build
SOURCEDIR   = .
BUILDDIR    = build
# https://github.com/sphinx-doc/sphinx/issues/8664#issuecomment-757087741
SPHINX_APIDOC_OPTIONS = members,show-inheritance

# Build the docs without the examples, which can take a long time to run
# From https://sphinx-gallery.github.io/stable/advanced.html#id4
html-noplot:
	@$(SPHINXBUILD) --builder html --define plot_gallery=0 "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
