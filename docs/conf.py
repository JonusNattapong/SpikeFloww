"""
Sphinx configuration for SpikeFlow documentation
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'SpikeFlow'
copyright = '2024, JonusNattapong'
author = 'JonusNattapong'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'myst_parser',
]

# Templates
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/spikeflow_logo.png'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Math support
mathjax3_config = {
    'tex': {'tags': 'ams', 'useLabelIds': True},
}
