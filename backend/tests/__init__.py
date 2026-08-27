"""Test suite for the FastAPI backend.

The backend modules import each other flatly -- ``from config import settings``
rather than ``from backend.config import settings`` -- so discovery has to run
with backend itself as the top-level directory:

    cd backend
    .venv-linux/bin/python -m unittest discover -s tests -t .

The ``-t .`` is what puts backend on sys.path, so a test can import the module
it exercises the same way the application does.
"""

from pathlib import Path

# The application modules, and the fake worker scripts these tests spawn as
# subprocesses, are resolved from here rather than from each test's own
# location.
BACKEND_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
