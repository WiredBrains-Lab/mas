Getting Started
===============

Prerequisites
-------------

- Python 3.9+ on Windows (recommended)
- Git (optional)

Setup
-----

Create and activate a virtual environment (optional but recommended):

.. code-block:: console

   python -m venv .venv
   .\.venv\Scripts\Activate

Install runtime dependencies (example minimal set):

.. code-block:: console

   pip install -r docs/requirements.txt

Run the App
-----------

From the repo root:

.. code-block:: console

   python single_neuron_analysis_suite.py

Build the Docs
--------------

.. code-block:: console

   python -m sphinx -M html docs/source docs/build

Open ``docs/build/html/index.html``.

Notes
-----

- If PyQt or other GUI deps aren't installed, the docs still build due to
  mocked imports (see ``autodoc_mock_imports`` in ``docs/source/conf.py``).
