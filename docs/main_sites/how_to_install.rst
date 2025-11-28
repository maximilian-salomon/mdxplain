How to Install mdxplain
=======================

Prerequisites
-------------

- Python >= 3.8 (Python 3.12 recommended)
- Virtual environment manager (conda or venv)

Quick Setup
-----------

mdxplain uses a Makefile for streamlined installation and development workflows.

.. code-block:: bash

  git clone https://github.com/maximilian-salomon/mdxplain.git
  cd mdxplain

Conda Environment (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a new conda environment with production dependencies, Jupyter, and nglview:

.. code-block:: bash

  make setup-conda
  conda activate mdxplain


Python Virtual Environment (Alternative)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a new virtual environment with production dependencies, Jupyter, and nglview:

.. code-block:: bash

  make setup-venv
  source mdxplain-venv/bin/activate

Install in Existing Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install mdxplain core package in your currently active environment (without Jupyter
and nglview):

.. code-block:: bash

  make install

To add Jupyter and nglview to an existing installation:

.. code-block:: bash

  make install-jupyter

Optional: PyMOL Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyMOL is not included in the standard installation to avoid compatibility issues due
to its complex system-level dependencies. If you need PyMOL for 3D structure
visualization, install it separately:

.. code-block:: bash

  make install-pymol

**Note:** For system-specific installation instructions, please refer to the
`official PyMOL documentation <https://pymolwiki.org/index.php/Linux_Install>`.
Alternatively, you can install PyMOL independently and load mdxplain's generated
PyMOL scripts manually.

Installation Options
--------------------

Environment Setup Commands
^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------+-------------+--------------------------------------------------------+
| Command                   | Environment | Description                                            |
+===========================+=============+========================================================+
| ``make setup-conda``      | conda       | Production dependencies with Jupyter and nglview       |
|                           |             | (no PyMOL)                                             |
+---------------------------+-------------+--------------------------------------------------------+
| ``make setup-dev-conda``  | conda       | Development dependencies with Jupyter and nglview      |
|                           |             | (no PyMOL)                                             |
+---------------------------+-------------+--------------------------------------------------------+
| ``make setup-pymol-conda``| conda       | Production dependencies with Jupyter, nglview and PyMOL|
+---------------------------+-------------+--------------------------------------------------------+
| ``make setup-venv``       | venv        | Production dependencies with Jupyter and nglview       |
|                           |             | (no PyMOL)                                             |
+---------------------------+-------------+--------------------------------------------------------+
| ``make setup-dev-venv``   | venv        | Development dependencies with Jupyter and nglview      |
|                           |             | (no PyMOL)                                             |
+---------------------------+-------------+--------------------------------------------------------+
| ``make setup-pymol-venv`` | venv        | Production dependencies with Jupyter, nglview and PyMOL|
+---------------------------+-------------+--------------------------------------------------------+

Installation in Existing Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------------------------+------------------------------------------------------------+
| Command                 | Description                                                |
+=========================+============================================================+
| ``make install``        | Install core package only (no Jupyter, no PyMOL)           |
+-------------------------+------------------------------------------------------------+
| ``make install-dev``    | Install with development dependencies plus Jupyter and     |
|                         | nglview (no PyMOL)                                         |
+-------------------------+------------------------------------------------------------+
| ``make install-jupyter``| Install Jupyter and nglview (adds to existing installation)|
+-------------------------+------------------------------------------------------------+
| ``make install-pymol``  | Install PyMOL for 3D structure visualization               |
+-------------------------+------------------------------------------------------------+
| ``make install-full``   | Install with full dependencies including Jupyter and PyMOL |
+-------------------------+------------------------------------------------------------+

What's Included
^^^^^^^^^^^^^^^

**Production Setup** (``make setup-conda`` / ``make setup-venv``)
  - Core mdxplain package
  - Jupyter ecosystem (JupyterLab, notebook, ipykernel, ipywidgets)
  - nglview for interactive 3D visualization in Jupyter
  - DPA (Density Peak Algorithm)

**Core Installation** (``make install``)
  - Core mdxplain package with basic dependencies
  - DPA (Density Peak Algorithm)
  - No Jupyter, no nglview, no PyMOL
  - Use this for minimal installations or server environments

**Development Setup** (``make setup-dev-conda`` / ``make setup-dev-venv`` / ``make install-dev``)
  - All production dependencies (core + Jupyter + nglview, no PyMOL)
  - Code formatting and style tools (black, autopep8, isort)
  - Linting and analysis tools (flake8, pylint, mypy, pydocstyle, etc.)
  - Testing and coverage tools (pytest, coverage)
  - Documentation tools (sphinx, sphinx-rtd-theme, myst-nb)
  - Security and profiling tools (bandit, safety, memory-profiler, snakeviz)

**PyMOL Setup** (``make setup-pymol-conda`` / ``make setup-pymol-venv`` / ``make install-pymol``)
  - Production dependencies with Jupyter, nglview and PyMOL
  - PyMOL for advanced 3D structure visualization (pymol-open-source >= 3.2.0a0)
  - Separate installation option to avoid compatibility issues
  - PyMOL only works in terminal/script environments (not Jupyter notebooks)
  - nglview provides 3D visualization within Jupyter notebooks

**Jupyter Addition** (``make install-jupyter``)
  - Jupyter ecosystem (JupyterLab, notebook, ipykernel, ipywidgets)
  - nglview for interactive 3D visualization
  - Adds Jupyter to existing mdxplain installation (e.g., after ``make install``)
  - Use this if you installed core only but now need Jupyter support

**Complete Installation** (``make install-full``)
  - Full installation with all dependencies in existing environment
  - Core mdxplain package
  - Jupyter ecosystem (JupyterLab, notebook, ipykernel, ipywidgets)
  - nglview for interactive 3D visualization in Jupyter
  - PyMOL for advanced 3D structure visualization

Additional Commands
-------------------

Jupyter
^^^^^^^

.. code-block:: bash

    make jupyter       # Start JupyterLab
    make notebook      # Start classic Jupyter Notebook

Development
^^^^^^^^^^^

.. code-block:: bash

    make test          # Run tests with pytest
    make lint          # Run code quality checks
    make format        # Format code with black and isort
    make html          # Build documentation
    make clean         # Remove environments and cache files

Help
^^^^

To see all available commands:

.. code-block:: bash

    make help

Install on Windows
------------------

Windows requires **Make** to be installed for using the Makefile commands. If you don't
have Make, you can install it via `Chocolatey <https://chocolatey.org/>`_ or use the
manual pip/conda commands below.

All make command except PyMOL installation are supported on Windows.

PyMOL Installation on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``make install-pymol`` command is not supported on Windows due to PyMOL's
GUI-specific dependencies and the need for pre-built wheel files. Instead, install
PyMOL manually using pre-compiled wheels.

**Installation Steps:**

Download and install the appropriate PyMOL wheel for your Python version from the
`official releases <https://github.com/cgohlke/pymol-open-source-wheels/releases>`_:

.. code-block:: powershell

   # Example for Python 3.12
   wget https://github.com/cgohlke/pymol-open-source-wheels/releases/download/v2025.10.30/pymol-3.2.0a0-cp312-cp312-win_amd64.whl
   pip install pymol-3.2.0a0-cp312-cp312-win_amd64.whl

**Note:**
 - Make sure to select the correct wheel file matching your Python version
   (e.g., ``cp312`` for Python 3.12, ``cp311`` for Python 3.11).
 - PyMOL version **3.2.0a0 or higher** is required for compatibility with numpy 2.x.
 - For detailed instructions and troubleshooting, visit the
   `PyMOL Windows Wheels repository <https://github.com/cgohlke/pymol-open-source-wheels>`_.

Manual Installation (without Make)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create a virtual environment:

   .. code-block:: powershell

      # Using conda
      conda create -n mdxplain python=3.12 -y
      conda activate mdxplain

      # Using venv
      python -m venv mdxplain-venv
      mdxplain-venv\Scripts\activate

2. Install mdxplain with desired dependencies:

   .. code-block:: powershell

      # Production with Jupyter and nglview (no PyMOL)
      pip install .[jupyter]
      pip install --no-deps DPA

      # Development with Jupyter and nglview (no PyMOL)
      pip install .[dev,jupyter]
      pip install --no-deps DPA

3. Install PyMOL separately:

  .. code-block:: powershell

    # Example for Python 3.12
    wget https://github.com/cgohlke/pymol-open-source-wheels/releases/download/v2025.10.30/pymol-3.2.0a0-cp312-cp312-win_amd64.whl
    pip install pymol-3.2.0a0-cp312-cp312-win_amd64.whl