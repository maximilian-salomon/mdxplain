# Makefile for Sphinx documentation

SPHINXBUILD = sphinx-build
SOURCEDIR = .
BUILDDIR = build

.PHONY: html clean

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(0)

clean:
	@rm -rf "$(BUILDDIR)"