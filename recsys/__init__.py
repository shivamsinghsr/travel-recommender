"""Recommendation core shared by the API, the training job and the web export.

Nothing in this package knows about HTTP or databases: it works on plain
Python objects and NumPy arrays, so it can be tested in isolation.
"""

__version__ = "1.0.0"
