import os
import sys
import tomllib
from pathlib import Path

# Read version from pyproject.toml
pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'
with open(pyproject_path, 'rb') as f:
    pyproject_data = tomllib.load(f)
    version = pyproject_data['project']['version']
    release = version

# Project information
project = 'mdxplain'
copyright = '2025, Maximilian Salomon'
contributers = [
    {'name': 'Maximilian Salomon', 'role': 'Software'},
    {'name': 'Maeve Branwen Butler', 'role': 'Documentation'},
]
author = 'author list from paper'

# Set path for root folder and source code folder
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../mdxplain'))

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'myst_nb',              # Exension for embedding Jupyter notebooks
]

# Sort order of methods/classes/attributes in documentation by source code
autodoc_member_order = 'bysource'

exclude_patterns = [
    '_build',
    'build',
    'jupyter_execute',
    '**/.ipynb_checkpoints',
]

# Substitution for a literal backslash ("\ |bsol| ") in docstrings to allow "\" in build
rst_prolog = """
.. |bsol| unicode:: 0x005C
    :trim:
"""

# Do not execute notebooks during doc builds (avoid sandbox/network/kernel issues)
nb_execution_mode = "off"

# html build configurations
# Emit docs/_build/html/index.html for ReadTheDocs root serving
root_doc = master_doc = 'index'
html_theme = 'sphinx_rtd_theme'
html_search = True
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 8,
    'titles_only': False,
}
