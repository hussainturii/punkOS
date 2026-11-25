import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from core.file_intel import FileSearchEngine
from PyQt6.QtWidgets import QApplication
from ui.file_search_view import FileSearchPanel, IndexWorker, SearchWorker


class FileSearchErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_partial_index_is_visible_then_retry_indexes_only_repaired_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'docs'
            folder.mkdir()
            (folder / 'good.txt').write_text('A valid synthetic document for indexing.')
            bad = folder / 'bad.docx'
            bad.write_bytes(b'not a Word archive')
            panel = FileSearchPanel(str(Path(tmp) / 'index.db'), [str(folder)])
            panel.engine._model = Mock()
            panel.engine._model.encode.return_value = np.ones(384, dtype='float32') / np.sqrt(384)
            worker = IndexWorker(panel.engine, [str(folder)])
            panel.worker = worker
            worker.error_signal.connect(panel._on_operation_error)
            worker.done_signal.connect(panel._on_index_done)
            worker.run()
            self.assertEqual(panel.engine.file_count(), 1)
            self.assertIn('Index incomplete. 1 new file(s)', panel.index_log.toPlainText())
            self.assertIn('bad.docx', panel.index_log.toPlainText())
            self.assertTrue(panel.index_btn.isEnabled())
            import docx
            doc = docx.Document()
            doc.add_paragraph('The repaired synthetic document is now readable.')
            doc.save(bad)
            panel._operation_error = ''
            panel.index_log.clear()
            worker.run()
            self.assertEqual(panel.engine.file_count(), 2)
            self.assertIn('Done. 1 new file(s)', panel.index_log.toPlainText())
            self.assertEqual(panel.engine.index_directory(folder), 0)
            panel.close()

    def test_missing_directory_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = FileSearchEngine(Path(tmp) / 'index.db')
            worker = IndexWorker(engine, [str(Path(tmp) / 'missing')])
            errors, counts = [], []
            worker.error_signal.connect(errors.append)
            worker.done_signal.connect(counts.append)
            worker.run()
            self.assertIn('missing or not a folder', errors[0])
            self.assertEqual(counts, [0])

    def test_unreadable_file_is_an_error_not_an_unchanged_skip(self):
        from core.file_intel import IndexingError
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'docs'
            folder.mkdir()
            (folder / 'locked.txt').write_text('A synthetic document whose read is denied.')
            engine = FileSearchEngine(Path(tmp) / 'index.db')
            with patch.object(engine, '_hash_file', side_effect=PermissionError('read denied')):
                with self.assertRaises(IndexingError) as raised:
                    engine.index_directory(folder)
            self.assertEqual(raised.exception.indexed, 0)
            self.assertIn('read denied', str(raised.exception))

    def test_cleanup_failure_reports_error_and_finishes(self):
        engine = Mock()
        engine.remove_missing.side_effect = RuntimeError('database locked')
        worker = IndexWorker(engine, ['fixture'])
        events = []
        worker.error_signal.connect(lambda message: events.append(('error', message)))
        worker.done_signal.connect(lambda count: events.append(('done', count)))
        worker.run()
        self.assertEqual(events, [('error', 'Index cleanup failed: database locked'), ('done', 0)])
        engine.index_directory.assert_not_called()

    def test_search_error_is_visible_and_controls_recover(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = FileSearchPanel(str(Path(tmp) / 'index.db'), [tmp])
            engine = Mock()
            engine.search.side_effect = RuntimeError('embedding unavailable')
            worker = SearchWorker(engine, 'fixture')
            panel.worker = worker
            panel.search_btn.setEnabled(False)
            panel.index_btn.setEnabled(False)
            worker.error_signal.connect(panel._on_operation_error)
            worker.done_signal.connect(panel._on_search_done)
            worker.run()
            self.assertIn('embedding unavailable', panel.results_list.item(0).text())
            self.assertNotIn('No matching', panel.results_list.item(0).text())
            self.assertTrue(panel.search_btn.isEnabled())
            self.assertTrue(panel.index_btn.isEnabled())
            panel.close()

    def test_repeated_submission_preserves_active_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = FileSearchPanel(str(Path(tmp) / 'index.db'), [tmp])
            active_thread, active_worker = Mock(), Mock()
            active_thread.isRunning.return_value = True
            panel.thread, panel.worker = active_thread, active_worker
            panel.query_input.setText('another query')
            panel._on_search_clicked()
            panel._on_index_clicked()
            self.assertIs(panel.thread, active_thread)
            self.assertIs(panel.worker, active_worker)
            panel.thread = panel.worker = None
            panel.close()


if __name__ == '__main__':
    unittest.main()
