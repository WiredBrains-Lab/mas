Developer Guide
===============

This project uses Sphinx with NumPy-style docstrings (Napoleon) to build
contributor-friendly API documentation.

Docstrings
----------

- Use a one-line summary, followed by a blank line and details.
- Include clear parameter types and shapes (e.g., ``ndarray, shape (n, m)``).
- List Returns and Raises where relevant.
- Provide a short example when it adds clarity.

Building Docs
-------------

.. code-block:: console

   python -m venv .venv
   .\.venv\Scripts\Activate
   pip install -r docs/requirements.txt
   python -m sphinx -M html docs/source docs/build

Open ``docs/build/html/index.html`` in your browser.

Heavy Dependencies
------------------

We mock heavy/GUI dependencies during autodoc to keep builds fast and
headless. See ``autodoc_mock_imports`` in ``docs/source/conf.py``. If you
add a new heavy dependency (e.g., a GUI toolkit), add its top-level import
name to this list.

Style and Linting
-----------------

- Prefer small, focused functions and clear names.
- Keep UI code on the main thread; offload heavy work to workers.
- Avoid changing behavior in docstring-only commits.

