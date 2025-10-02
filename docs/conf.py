import os
import sys

project = 'mdxplain'
copyright = '2025, Max'
author = 'Max'
release = '01.11.2025'

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../mdxplain'))

extensions = [
    'sphinx.ext.autodoc',
]

# autoclass_content = 'both'
autodoc_member_order = 'bysource'
autodoc_mock_imports = ['Pipeline']  #findet beim Build nicht 'Pipeline' aus mdxplain/clustering/cluster_type/dpa/dpa_calculator.py

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
