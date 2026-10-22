.. MAS documentation master file, created by
   sphinx-quickstart on Sun Jun 14 12:01:20 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MEG Analysis Suite
==================

An application for MEG and multimodal data analysis, visualization,
and neural concept decoding. Explore the usage guide and API to get started.

.. note::
   This project is under active development.

Quick Start
-----------

Build the documentation locally (no PATH needed):

1. Create/activate a virtual environment (optional):

   - Windows::

       python -m venv .venv
       .\\.venv\\Scripts\\Activate

2. Install docs requirements::

       pip install -r docs/requirements.txt

3. Build HTML docs::

       python -m sphinx -M html docs/source docs/build

Open ``docs/build/html/index.html`` in your browser.

Contributing Docs
-----------------

- Docstrings use the NumPy style and are rendered by Sphinx Napoleon.
- Prefer concise one‑line summaries, clear parameter types/shapes, and
  brief examples where helpful.
- Heavy/GUI dependencies are mocked during autodoc. See
  ``autodoc_mock_imports`` in ``docs/source/conf.py``.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage
   developer
   examples
   contributing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
   modules


Indices and Tables
------------------

- General index: :ref:`genindex`
- Module index: :ref:`modindex`
- Search: :ref:`search`
