# Configuration file for the Sphinx documentation builder.
# sphinx-build -M html docs/source/ docs/build/
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-source_suffix
import sys
import os
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MAS'
copyright = '2026, Wired Brains Lab, Department of Neurosurgery, Medical College of Wisconsin'
author = 'Sunil Mathew'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Napoleon / Autodoc options ---------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'inherited-members': True,
}

# Mock optional heavy dependencies so autodoc can build even without them
autodoc_mock_imports = [
    'torch', 'torchvision', 'transformers', 'diffusers',
    'SimpleITK', 'gensim', 'pyarrow',
]

# Generate autosummary pages automatically
autosummary_generate = True

