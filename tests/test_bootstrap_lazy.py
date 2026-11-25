import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.bootstrap import build_subsystems


class BootstrapLazyTests(unittest.TestCase):
    def test_startup_does_not_load_optional_embedding_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('core.embedding.TextEmbedder._load') as load:
                subs = build_subsystems({}, project_root=Path(directory))
                self.assertIsNotNone(subs.memory)
                load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
