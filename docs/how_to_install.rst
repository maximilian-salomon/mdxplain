How to Install mdxplain
=======================

Prerequisites
-------------

- Python >= 3.8 (Python 3.12 recommended)
- Virtual environment manager (venv or conda)

Quick Setup
-----------

mdxplain uses a Makefile for streamlined installation and development workflows:

.. code-block::

    # Clone repository
    git clone https://github.com/BioFreak95/mdxplain.git
    cd mdxplain

    # Full setup with development tools and Jupyter (recommended)
    make setup-full-env
    source mdxplain-venv/bin/activate  # Linux/Mac

    # Or using conda
    make setup-conda-full-env
    conda activate mdxplain

Installation Options
--------------------

+--------------------------------+-------------+--------------------------------------+
| Command                        | Environment | Description                          |
+================================+=============+======================================+
| `make setup-env`               | venv        | Production dependencies only         |
+--------------------------------+-------------+--------------------------------------+
| `make setup-jupyter-env`       | venv        | Production + Jupyter                 |
+--------------------------------+-------------+--------------------------------------+
| `make setup-dev-env`           | venv        | Development tools (linting, testing) |
+--------------------------------+-------------+--------------------------------------+
| `make setup-full-env`          | venv        | Development + Jupyter (recommended)  |
+--------------------------------+-------------+--------------------------------------+
| `make setup-conda-env`         | conda       | Production dependencies only         |
+--------------------------------+-------------+--------------------------------------+
| `make setup-conda-jupyter-env` | conda       | Production + Jupyter                 |
+--------------------------------+-------------+--------------------------------------+
| `make setup-conda-dev-env`     | conda       | Development tools                    |
+--------------------------------+-------------+--------------------------------------+
| `make setup-conda-full-env`    | conda       | Development + Jupyter                |
+--------------------------------+-------------+--------------------------------------+

Development Commands
--------------------

.. code-block::

    make help          # Show all available commands
    make test          # Run tests
    make lint          # Run code quality checks
    make format        # Format code with black and isort
    make jupyter       # Start JupyterLab
    make html          # Make documentation html build
    make clean         # Remove environments and cache files