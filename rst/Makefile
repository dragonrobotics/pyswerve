# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pyswerve
SOURCEDIR     = .
BUILDDIR      = ../docs

# files to be ignored by autodoc
IGNORED_FILES := robot.py setup.py occupancy_grid_display.py occ_grid_profile.py

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%.py: ;
%.pyx: ;

autodoc: ../*.py ../*.pyx
	cd ../; python setup.py build_ext --inplace
	@sphinx-apidoc -f -o ./ ../ $(IGNORED_FILES)

.PHONY: autodoc help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile autodoc
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
