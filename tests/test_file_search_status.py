import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtCore import Qt
from core.file_intel import FileSearchEngine, declared_status
from ui.file_search_view import SearchWorker, ResultItem, FileSearchPanel


class StatusSearchTests(unittest.TestCase):
    def test_only_unambiguous_exact_fields_count(self):
        self.assertEqual(declared_status('Notes\nSTATUS: Approved\nOwner: Ana'), 'approved')
        for text in ('This is approved.', 'Status: Not approved', 'Status: Unapproved',
                     'Status: Approved pending review', 'Status: Approved\nStatus: Draft',
                     'Status: Approved\nStatus: Unknown', 'Approval status: Approved'):
            with self.subTest(text=text):
                self.assertIsNone(declared_status(text))

    def test_filter_preserves_relevance_scope_and_original_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / 'docs'
            docs.mkdir()
            engine = FileSearchEngine(root / 'index.db')
            q = np.zeros(384, dtype='float32')
            q[0] = 1
            def encode(text, **kwargs):
                score = .1 if 'unrelated' in text else .8 if 'Status: Approved' in text else .9
                v = np.zeros(384, dtype='float32')
                v[0], v[1] = score, np.sqrt(1 - score * score)
                return v
            engine._model = Mock()
            engine._model.encode.side_effect = encode
            files = {
                'approved.txt': 'Meeting old proposal. Status is only in filename.\nStatus: Proposed',
                'neutral.txt': 'Meeting decision for Thursday.\nStatus: Approved',
                'comment.txt': 'Unapproved suggestion about an approved meeting decision.',
                'other.txt': 'unrelated purchasing decision.\nStatus: Approved',
                'ambiguous.txt': 'Meeting decision\nStatus: Approved\nStatus: Draft',
            }
            for name, content in files.items():
                (docs / name).write_text(content, encoding='utf-8')
            (root / 'outside.txt').write_text('Meeting decision outside scope.\nStatus: Approved')
            engine.index_directory(root)
            engine._model.encode.side_effect = None
            engine._model.encode.return_value = q
            hits = engine.search('meeting', path_filter=[str(docs)], status_filter='approved')
            self.assertEqual([Path(h.path).name for h in hits], ['neutral.txt'])
            self.assertAlmostEqual(hits[0].score, .8)
            self.assertEqual(hits[0].fields['declared_status'], 'approved')
            self.assertEqual([Path(h.path).name for h in engine.search(
                'meeting', path_filter=[str(docs)], status_filter='proposed')], ['approved.txt'])
            self.assertEqual(engine.search('meeting', status_filter='rejected'), [])
            all_hits = engine.search('meeting', path_filter=[str(docs)])
            self.assertEqual([h.score for h in all_hits], sorted([h.score for h in all_hits], reverse=True))
            self.assertEqual(engine.search('meeting', path_filter=[], status_filter='approved'), [])
            with self.assertRaises(ValueError):
                engine.search('meeting', status_filter='not approved')

    def test_worker_and_result_card_expose_declared_status(self):
        app = QApplication.instance() or QApplication([])
        engine = Mock()
        engine.search.return_value = []
        worker = SearchWorker(engine, 'meeting', ['docs'], 'approved')
        worker.run()
        engine.search.assert_called_once_with('meeting', path_filter=['docs'], status_filter='approved')
        from core.file_intel import FileResult
        card = ResultItem(FileResult('neutral.txt', .8, 'Meeting decision', 'Note',
                                     {'declared_status': 'approved'}, ''))
        labels = [label.text() for label in card.findChildren(QLabel)]
        self.assertIn('Declared status: approved', labels)
        self.assertIn('Similarity 0.80', labels)
        card.close()

    def test_changing_filter_clears_previous_results(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            panel = FileSearchPanel(str(Path(tmp) / 'index.db'), [tmp])
            panel.results_list.addItem('Old proposed document')
            panel.status_filter.setCurrentIndex(panel.status_filter.findData('approved'))
            self.assertEqual(panel.results_list.count(), 1)
            self.assertIn('Press Search', panel.results_list.item(0).text())
            panel.status_filter.setEnabled(False)
            panel._operation_error = 'Search failed: synthetic error'
            panel._on_search_done([])
            self.assertTrue(panel.status_filter.isEnabled())
            self.assertIn('synthetic error', panel.results_list.item(0).text())
            panel.close()

    def test_rendered_results_expose_content_and_replace_stale_accessible_text(self):
        app = QApplication.instance() or QApplication([])
        from core.file_intel import FileResult
        with tempfile.TemporaryDirectory() as tmp:
            panel = FileSearchPanel(str(Path(tmp) / 'index.db'), [tmp])
            row = FileResult(str(Path(tmp) / 'invoice.txt'), .65, 'Total: $2400.00',
                             'Invoice', {'declared_status': 'approved', 'amount': '$2400.00',
                                         'vendor': 'Cedar'}, '')
            panel._on_search_done([row])
            item = panel.results_list.item(0)
            summary = item.data(Qt.ItemDataRole.AccessibleTextRole)
            for expected in ('invoice.txt', row.path, 'Similarity: 0.65',
                             'Declared status: approved', 'Amount: $2400.00',
                             'Vendor: Cedar', 'Content excerpt: Total: $2400.00'):
                self.assertIn(expected, summary)
            self.assertEqual(item.data(Qt.ItemDataRole.UserRole), row.path)
            self.assertEqual(item.text(), '')  # no duplicate text painted behind the card
            panel._on_search_done([])
            self.assertEqual(panel.results_list.count(), 1)
            self.assertNotIn('invoice.txt', str(panel.results_list.item(0).data(Qt.ItemDataRole.AccessibleTextRole)))
            panel.close()


if __name__ == '__main__':
    unittest.main()
