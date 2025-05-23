# Minimal Sphinx Makefile for HTML/PDF builds

SPHINXBUILD   ?= python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html

help:
	@echo "Sphinx documentation make targets:"
	@echo "  make html    # build HTML docs"
	@echo "  make clean   # remove build artifacts"

clean:
	$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"