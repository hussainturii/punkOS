import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from core.file_intel import FileSearchEngine


class FileSearchScopeTests(unittest.TestCase):
    def test_home_expansion_and_directory_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            docs, sibling = home / 'Documents', home / 'Documents-old'
            docs.mkdir()
            sibling.mkdir()
            (docs / 'allowed.txt').write_text('approved meeting decision for the fixture', encoding='utf-8')
            (sibling / 'excluded.txt').write_text('approved meeting decision for the fixture', encoding='utf-8')
            engine = FileSearchEngine(home / 'index.db')
            engine._model = Mock()
            engine._model.encode.return_value = np.ones(384, dtype='float32') / np.sqrt(384)
            with patch.dict(os.environ, {'USERPROFILE': tmp, 'HOME': tmp}):
                self.assertEqual(engine.index_directory('~/Documents'), 1)
                self.assertEqual(engine.index_directory(sibling), 1)
                hits = engine.search('approved', path_filter=['~/Documents'])
                self.assertEqual([Path(h.path).name for h in hits], ['allowed.txt'])
                self.assertEqual(engine.search('approved', path_filter=[]), [])
                self.assertEqual(engine.search('approved', path_filter=['']), [])
                self.assertEqual(len(engine.search('approved', path_filter=None)), 2)


if __name__ == '__main__':
    unittest.main()
