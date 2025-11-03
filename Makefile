.PHONY: help setup-env setup-jupyter-env setup-dev-env setup-full-env setup-conda-env setup-conda-jupyter-env setup-conda-dev-env setup-conda-full-env install install-dev install-jupyter install-visualization install-full test lint format jupyter notebook html clean

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Environment Setup (Python venv):"
	@echo "  setup-env           Create venv with production dependencies"
	@echo "  setup-jupyter-env   Create venv with production + jupyter dependencies"
	@echo "  setup-dev-env       Create venv with development dependencies"
	@echo "  setup-full-env      Create venv with development + jupyter + visualization"
	@echo ""
	@echo "Environment Setup (Conda):"
	@echo "  setup-conda-env         Create conda env with production dependencies"
	@echo "  setup-conda-jupyter-env Create conda env with production + jupyter dependencies"
	@echo "  setup-conda-dev-env     Create conda env with development dependencies"
	@echo "  setup-conda-full-env    Create conda env with dev + jupyter + visualization"
	@echo ""
	@echo "Installation (in existing environment):"
	@echo "  install             Install package in current environment"
	@echo "  install-dev         Install package with dev dependencies"
	@echo "  install-jupyter     Install package with jupyter dependencies"
	@echo "  install-visualization Install package with visualization dependencies (PyMOL)"
	@echo "  install-full        Install package with dev + jupyter + visualization"
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

# Create fresh virtual environment with production dependencies only
setup-env:
	@echo "Creating new virtual environment 'mdxplain-venv'..."
	python -m venv mdxplain-venv
	@echo "Upgrading pip..."
	mdxplain-venv/bin/pip install --upgrade pip
	@echo "Installing production dependencies..."
	mdxplain-venv/bin/pip install .
	@echo "Installing DPA with --no-deps..."
	mdxplain-venv/bin/pip install --no-deps DPA
	@echo "Setup complete! Activate with: source mdxplain-venv/bin/activate"

# Create fresh virtual environment with production + jupyter dependencies
setup-jupyter-env:
	@echo "Creating new virtual environment 'mdxplain-venv' with Jupyter..."
	python -m venv mdxplain-venv
	@echo "Upgrading pip..."
	mdxplain-venv/bin/pip install --upgrade pip
	@echo "Installing production + jupyter dependencies..."
	mdxplain-venv/bin/pip install .[jupyter]
	@echo "Installing DPA with --no-deps..."
	mdxplain-venv/bin/pip install --no-deps DPA
	@echo "Jupyter setup complete! Activate with: source mdxplain-venv/bin/activate"

# Create fresh virtual environment with development dependencies
setup-dev-env:
	@echo "Creating new development virtual environment 'mdxplain-venv'..."
	python -m venv mdxplain-venv
	@echo "Upgrading pip..."
	mdxplain-venv/bin/pip install --upgrade pip
	@echo "Installing development dependencies..."
	mdxplain-venv/bin/pip install .[dev]
	@echo "Installing DPA with --no-deps..."
	mdxplain-venv/bin/pip install --no-deps DPA
	@echo "Installing pre-commit hooks..."
	mdxplain-venv/bin/pre-commit install
	@echo "Development setup complete! Activate with: source mdxplain-venv/bin/activate"

# Create fresh virtual environment with development + jupyter + visualization dependencies
setup-full-env:
	@echo "Creating new full virtual environment 'mdxplain-venv'..."
	python -m venv mdxplain-venv
	@echo "Upgrading pip..."
	mdxplain-venv/bin/pip install --upgrade pip
	@echo "Installing development + jupyter + visualization dependencies..."
	mdxplain-venv/bin/pip install .[dev,jupyter,visualization]
	@echo "Installing DPA with --no-deps..."
	mdxplain-venv/bin/pip install --no-deps DPA
	@echo "Installing pre-commit hooks..."
	mdxplain-venv/bin/pre-commit install
	@echo "Full setup complete! Activate with: source mdxplain-venv/bin/activate"

# Install package in current environment
install:
	pip install .
	pip install --no-deps DPA

# Install package with development dependencies in current environment
install-dev:
	pip install .[dev]
	pip install --no-deps DPA
	pre-commit install

# Install package with jupyter dependencies in current environment
install-jupyter:
	pip install .[jupyter]
	pip install --no-deps DPA

# Install package with visualization dependencies in current environment
install-visualization:
	pip install .[visualization]
	pip install --no-deps DPA

# Install package with development, jupyter and visualization dependencies in current environment
install-full:
	pip install .[dev,jupyter,visualization]
	pip install --no-deps DPA

# Create fresh conda environment with production dependencies only
setup-conda-env:
	@echo "Creating new conda environment 'mdxplain'..."
	conda create -n mdxplain python=3.12 -y
	@echo "Installing pip in conda environment..."
	conda run -n mdxplain conda install pip -y
	@echo "Installing production dependencies..."
	conda run -n mdxplain pip install .
	@echo "Installing DPA with --no-deps..."
	conda run -n mdxplain pip install --no-deps DPA
	@echo "Setup complete! Activate with: conda activate mdxplain"

# Create fresh conda environment with production + jupyter dependencies
setup-conda-jupyter-env:
	@echo "Creating new conda environment 'mdxplain' with Jupyter..."
	conda create -n mdxplain python=3.12 -y
	@echo "Installing pip in conda environment..."
	conda run -n mdxplain conda install pip -y
	@echo "Installing production + jupyter dependencies..."
	conda run -n mdxplain pip install .[jupyter]
	@echo "Installing DPA with --no-deps..."
	conda run -n mdxplain pip install --no-deps DPA
	@echo "Jupyter setup complete! Activate with: conda activate mdxplain"

# Create fresh conda environment with development dependencies
setup-conda-dev-env:
	@echo "Creating new conda environment 'mdxplain'..."
	conda create -n mdxplain python=3.12 -y
	@echo "Installing pip in conda environment..."
	conda run -n mdxplain conda install pip -y
	@echo "Installing development dependencies..."
	conda run -n mdxplain pip install .[dev]
	@echo "Installing DPA with --no-deps..."
	conda run -n mdxplain pip install --no-deps DPA
	@echo "Installing pre-commit hooks..."
	conda run -n mdxplain pre-commit install
	@echo "Development setup complete! Activate with: conda activate mdxplain"

# Create fresh conda environment with development + jupyter + visualization dependencies
setup-conda-full-env:
	@echo "Creating new conda environment 'mdxplain' with full setup..."
	conda create -n mdxplain python=3.12 -y
	@echo "Installing pip in conda environment..."
	conda run -n mdxplain conda install pip -y
	@echo "Installing development + jupyter + visualization dependencies..."
	conda run -n mdxplain pip install .[dev,jupyter,visualization]
	@echo "Installing DPA with --no-deps..."
	conda run -n mdxplain pip install --no-deps DPA
	@echo "Installing pre-commit hooks..."
	conda run -n mdxplain pre-commit install
	@echo "Full setup complete! Activate with: conda activate mdxplain"

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


# Set directions for docs build.
SPHINXBUILD = sphinx-build
SOURCEDIR = docs
BUILDDIR = docs/build

# Sphinx html build.
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(0)

# Clean up
clean:
	@echo "Removing virtual environment..."
	rm -rf mdxplain-venv
	@echo "Removing conda environment (if exists)..."
	-conda env remove -n mdxplain -y
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