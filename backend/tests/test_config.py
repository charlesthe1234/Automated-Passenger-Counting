import os
import sys
import tempfile
import unittest
from pathlib import Path


from tests import BACKEND_DIR
sys.path.insert(0, str(BACKEND_DIR))

from config import Settings


class SettingsPathTests(unittest.TestCase):
    def test_worker_python_path_does_not_resolve_virtualenv_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base_python = directory / "base-python"
            base_python.touch()
            virtualenv_python = directory / "venv-python"
            virtualenv_python.symlink_to(base_python)

            settings = Settings(CV_WORKER_PYTHON=str(virtualenv_python), _env_file=None)

            self.assertEqual(
                settings.cv_worker_python_path,
                Path(os.path.abspath(virtualenv_python)),
            )
            self.assertNotEqual(settings.cv_worker_python_path, base_python.resolve())


if __name__ == "__main__":
    unittest.main()
