.PHONY: help setup-conda setup-dev-conda setup-pymol-conda setup-venv setup-dev-venv setup-pymol-venv install install-dev install-jupyter install-pymol install-full test lint format jupyter notebook html clean

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Environment Setup (Conda):"
	@echo "  setup-conda         Create conda env with production dependencies (without PyMOL)"
	@echo "  setup-dev-conda     Create conda env with development dependencies (without PyMOL)"
	@echo "  setup-pymol-conda   Create conda env with jupyter and PyMOL"
	@echo ""
	@echo "Environment Setup (Python venv):"
	@echo "  setup-venv          Create venv with production dependencies (without PyMOL)"
	@echo "  setup-dev-venv      Create venv with development dependencies (without PyMOL)"
	@echo "  setup-pymol-venv    Create venv with jupyter and PyMOL"
	@echo ""
	@echo "Installation (in existing environment):"
	@echo "  install             Install package (without PyMOL)"
	@echo "  install-dev         Install package with development dependencies (without PyMOL)"
	@echo "  install-jupyter     Install Jupyter dependencies"
	@echo "  install-pymol       Install PyMOL for 3D structure visualization"
	@echo "  install-full        Install package with full dependencies including Jupyter and PyMOL"
	@echo ""
	@echo "Jupyter:"
	@echo "  jupyter             Start JupyterLab"
	@echo "  notebook            Start classic Jupyter Notebook"
	@echo ""
	@echo "Development:"
	@echo "  test                Run tests with pytest"
	@echo "  lint                Run code quality checks"
	@echo "  format              Format code with black and isort"
	@echo "  clean               Remove environments and cache files"

# Create fresh conda environment with production dependencies
setup-conda:
	@echo "Creating new conda environment 'mdxplain'..."
	conda create -n mdxplain python=3.12 -y
	@echo "Installing pip in conda environment..."
	conda run -n mdxplain conda install pip -y
	@echo "Installing production dependencies with jupyter..."
	conda run -n mdxplain pip install .[jupyter]
	@echo "Installing DPA with --no-deps..."
	conda run -n mdxplain pip install --no-deps DPA
	@echo "Setup complete! Activate with: conda activate mdxplain"

# Create fresh conda environment with development dependencies
setup-dev-conda:
	@echo "Creating new conda environment 'mdxplain' for development..."
	conda create -n mdxplain python=3.12 -y
	@echo "Installing pip in conda environment..."
	conda run -n mdxplain conda install pip -y
	@echo "Installing development dependencies..."
	conda run -n mdxplain pip install .[dev,jupyter]
	@echo "Installing DPA with --no-deps..."
	conda run -n mdxplain pip install --no-deps DPA
	@echo "Development setup complete! Activate with: conda activate mdxplain"

# Create fresh conda environment with jupyter and pymol
setup-pymol-conda:
	@echo "Creating new conda environment 'mdxplain' with jupyter and pymol..."
	conda create -n mdxplain python=3.12 -y
	@echo "Installing pip in conda environment..."
	conda run -n mdxplain conda install pip -y
	@echo "Installing jupyter and pymol..."
	conda run -n mdxplain pip install .[jupyter,pymol]
	@echo "Installing DPA with --no-deps..."
	conda run -n mdxplain pip install --no-deps DPA
	@echo "Development setup complete! Activate with: conda activate mdxplain"

# Create fresh virtual environment with production dependencies only
setup-venv:
	@echo "Creating new virtual environment 'mdxplain-venv'..."
	python -m venv mdxplain-venv
	@echo "Upgrading pip..."
	mdxplain-venv/bin/pip install --upgrade pip
	@echo "Installing production dependencies..."
	mdxplain-venv/bin/pip install .[jupyter]
	@echo "Installing DPA with --no-deps..."
	mdxplain-venv/bin/pip install --no-deps DPA
	@echo "Setup complete! Activate with: source mdxplain-venv/bin/activate"

# Create fresh virtual environment with development dependencies
setup-dev-venv:
	@echo "Creating new development virtual environment 'mdxplain-venv'..."
	python -m venv mdxplain-venv
	@echo "Upgrading pip..."
	mdxplain-venv/bin/pip install --upgrade pip
	@echo "Installing development dependencies..."
	mdxplain-venv/bin/pip install .[dev,jupyter]
	@echo "Installing DPA with --no-deps..."
	mdxplain-venv/bin/pip install --no-deps DPA
	@echo "Development setup complete! Activate with: source mdxplain-venv/bin/activate"

# Create fresh virtual environment with jupyter and pymol
setup-pymol-venv:
	@echo "Creating new virtual environment 'mdxplain-venv' with jupyter and pymol..."
	python -m venv mdxplain-venv
	@echo "Upgrading pip..."
	mdxplain-venv/bin/pip install --upgrade pip
	@echo "Installing jupyter and pymol..."
	mdxplain-venv/bin/pip install .[jupyter,pymol]
	@echo "Installing DPA with --no-deps..."
	mdxplain-venv/bin/pip install --no-deps DPA
	@echo "Development setup complete! Activate with: source mdxplain-venv/bin/activate"

# Install package in current environment
install:
	@echo "Installing mdxplain..."
	pip install .
	pip install --no-deps DPA
	@echo "Installation complete!"

# Install package with development dependencies in current environment
install-dev:
	@echo "Installing mdxplain with development dependencies..."
	pip install .[dev,jupyter]
	pip install --no-deps DPA
	@echo "Development installation complete!"

# Install Jupyter dependencies
install-jupyter:
	@echo "Installing Jupyter dependencies..."
	pip install .[jupyter]
	pip install --no-deps DPA
	@echo "Jupyter installation complete!"

# Install PyMOL separately
install-pymol:
	@echo "Installing PyMOL..."
	pip install .[pymol]
	pip install --no-deps DPA
	@echo "PyMOL installation complete!"

install-full:
	@echo "Installing mdxplain with full dependencies including Jupyter and PyMOL..."
	pip install .[dev,jupyter,pymol]
	pip install --no-deps DPA
	@echo "Full installation complete!"

# Run tests
test:
	python -m pytest

# Run code quality checks
lint:
	python -m flake8 mdxplain
	python -m pylint mdxplain
	python -m mypy mdxplain
	python -m bandit -r mdxplain
	python -m pydocstyle mdxplain --count
	python -m interrogate mdxplain --verbose --fail-under=100

# Format code
format:
	python -m black mdxplain
	python -m isort mdxplain

# Start JupyterLab
jupyter:
	jupyter lab

# Start classic Jupyter Notebook
notebook:
	jupyter notebook

# Set directions for docs build
SPHINXBUILD = sphinx-build
SOURCEDIR = docs
BUILDDIR = docs/build

# Prepare notebooks and build Sphinx html
html:
	@echo "Preparing notebooks for documentation..."
	python docs/dev/prepare_notebooks.py
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(0)

# Clean up
clean:
	@echo "Removing conda environment (if exists)..."
	-conda env remove -n mdxplain -y
	@echo "Removing virtual environment..."
	rm -rf mdxplain-venv
	@echo "Removing Python cache files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "Removing docs html build..."
	rm -rf "$(BUILDDIR)"
	@echo "Cleanup complete!"
