"""Test suite for the edge_tracker CV pipeline.

The pipeline modules import each other flatly -- ``from constants import ...``
rather than ``from edge_tracker.constants import ...`` -- because cv_worker.py
is launched by path rather than as an installed package. Discovery therefore
has to run with edge_tracker itself as the top-level directory:

    cd edge_tracker
    ../.venv-cv-linux/bin/python -m unittest discover -s tests -t .

The ``-t .`` is what puts edge_tracker on sys.path, so a test can import the
module it exercises the same way production code does.
"""

from pathlib import Path

# The pipeline modules and their data files (tracker configs, launcher_ui.py)
# live one level up. Tests that need to reach them use this rather than
# spelling out ``Path(__file__).parent.parent``, which stops meaning anything
# recognisable once you are reading it from inside a test.
EDGE_TRACKER_DIR = Path(__file__).resolve().parent.parent
