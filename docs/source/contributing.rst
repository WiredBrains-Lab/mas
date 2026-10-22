Contributing
============

Thanks for your interest in contributing! This guide outlines how to set up
your environment, coding conventions, and how to submit changes.

Workflow
--------

1. Fork and clone the repository.
2. Create a feature branch: ``git checkout -b feat/short-name``.
3. Make focused changes with clear commit messages.
4. Run the app locally and build docs (see Getting Started).
5. Open a Pull Request describing the change and motivation.

Coding Conventions
------------------

- Prefer small, composable functions and clear names.
- Keep UI operations on the main thread; use workers for heavy tasks.
- Use NumPy-style docstrings (see below) for all public functions/classes.
- Note array shapes and units (e.g., seconds, Hz, µV).
- Avoid changing behavior when editing docs unless necessary.

Docstrings (NumPy Style)
------------------------

Example function docstring:

.. code-block:: python

   def detect_spikes(data, distance=45, det_neg=True):
       """Detect spikes using robust thresholding.

       Parameters
       ----------
       data : ndarray
           1D voltage trace (µV) sampled uniformly.
       distance : int, default 45
           Minimum sample distance between detected spikes (refractory).
       det_neg : bool, default True
           Detect negative-going spikes when True; positive otherwise.

       Returns
       -------
       spikes : ndarray
           Indices of detected spikes.
       thr_min : float
           Detection threshold used.
       """

Build and Test Docs
-------------------

.. code-block:: console

   pip install -r docs/requirements.txt
   python -m sphinx -M html docs/source docs/build

Open ``docs/build/html/index.html`` to review changes. Heavy/GUI deps are
mocked during autodoc; add new heavy modules to ``autodoc_mock_imports`` in
``docs/source/conf.py`` if needed.

