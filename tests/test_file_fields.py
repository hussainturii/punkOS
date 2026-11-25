import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from core.file_intel import FileSearchEngine, extract_fields


class FileFieldTests(unittest.TestCase):
    def test_currency_does_not_truncate_unseparated_thousands(self):
        for text, expected in [('$1500.00', '$1500.00'), ('$1,500.00', '$1500.00'),
                               ('$125000', '$125000.00'), ('$9.50', '$9.50'),
                               ('$1,234,567.89', '$1234567.89'),
                               ('Deposit $200; total $1500.', '$1500.00')]:
            with self.subTest(text=text):
                self.assertEqual(extract_fields(text)['amount'], expected)

    def test_reindex_repairs_old_fields_without_reembedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'docs'
            folder.mkdir()
            (folder / 'invoice.txt').write_text('Invoice from Cedar. Total $1500.00')
            engine = FileSearchEngine(Path(tmp) / 'index.db')
            engine._model = Mock()
            engine._model.encode.return_value = np.ones(384, dtype='float32') / np.sqrt(384)
            self.assertEqual(engine.index_directory(folder), 1)
            with engine._conn() as conn:
                conn.execute('UPDATE files SET fields_json = ?', ('{"amount": "$150.00"}',))
            self.assertEqual(engine.index_directory(folder), 0)
            engine._model.encode.assert_called_once()
            with engine._conn() as conn:
                self.assertIn('$1500.00', conn.execute('SELECT fields_json FROM files').fetchone()[0])

    def test_rebuild_repairs_damaged_vector_for_unchanged_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'docs'
            folder.mkdir()
            (folder / 'invoice.txt').write_text('Invoice from Cedar. Total $1500.00')
            engine = FileSearchEngine(Path(tmp) / 'index.db')
            engine._model = Mock()
            engine._model.encode.return_value = np.ones(384, dtype='float32') / np.sqrt(384)
            engine.index_directory(folder)
            with engine._conn() as conn:
                conn.execute('UPDATE files SET embedding = ?', (b'broken',))
            with self.assertRaisesRegex(RuntimeError, 'Build / update index'):
                engine.search('invoice')
            self.assertEqual(engine.index_directory(folder), 1)
            self.assertEqual(len(engine.search('invoice')), 1)
