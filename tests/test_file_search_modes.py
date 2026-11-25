"""Isolated search fixtures; no real documents, downloads or API calls."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from core.file_intel import FileSearchEngine


class SearchModesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = FileSearchEngine(self.root/'index.db')
        self.engine._model = Mock()
        self.engine._model.encode.return_value = np.ones(384,dtype='float32')/np.sqrt(384)
        for folder in ('alpha','beta'):
            (self.root/folder).mkdir()
            (self.root/folder/'Report June.txt').write_text('Project review notes with confirmed delivery details.\n'+folder,encoding='utf-8')
        self.long = self.root/'alpha'/'notes.txt'
        self.long.write_text('Introductory background. '*140+'Unique turquoise marker at the end.',encoding='utf-8')
        self.engine.index_directory(self.root)
        self.engine._model.encode.reset_mock()

    def test_partial_name_keeps_duplicate_paths_and_needs_no_model(self):
        hits=self.engine.search('REPORT JUNE',search_mode='filename')
        self.assertEqual(len(hits),2)
        self.assertEqual(len({h.path for h in hits}),2)
        self.engine._model.encode.assert_not_called()

    def test_semantic_keyword_evidence_ranks_and_snippets_beyond_prefix(self):
        hits = self.engine.search('turquoise marker', top_k=1)
        self.assertEqual(hits[0].path, str(self.long.resolve()))
        self.assertIn('turquoise marker', hits[0].snippet)
        # The displayed score remains cosine similarity, not a fabricated
        # confidence boosted by the lexical evidence.
        self.assertAlmostEqual(hits[0].score, 1.0, places=5)

    def test_semantic_copies_do_not_crowd_out_distinct_documents(self):
        text = 'Identical delivery notes for a review.'
        for name in ('a.txt', 'b.txt', 'c.txt'):
            (self.root/name).write_text(text, encoding='utf-8')
        self.engine.index_directory(self.root)
        hits = self.engine.search('delivery review', top_k=4)
        self.assertEqual(len(hits), 4)
        self.assertEqual(len({Path(h.path).read_bytes() for h in hits}), 4)
        # Copies remain retrievable; no file/path is silently excluded.
        all_hits = self.engine.search('delivery review', top_k=20)
        self.assertEqual(len(all_hits), 6)

    def test_semantic_match_does_not_require_literal_query_words(self):
        hits = self.engine.search('unrelated vocabulary', top_k=10)
        self.assertEqual(len(hits), 3)

    def test_changed_copy_does_not_hide_unchanged_copy(self):
        original = self.root/'a.txt'
        duplicate = self.root/'b.txt'
        for path in (original, duplicate):
            path.write_text('Turquoise marker report', encoding='utf-8')
        self.engine.index_directory(self.root)
        original.write_text('Changed since indexing', encoding='utf-8')
        hits = self.engine.search('turquoise marker report', top_k=1)
        self.assertEqual(hits[0].path, str(duplicate.resolve()))

    def test_literal_content_uses_stored_text_beyond_embedding_prefix(self):
        hits=self.engine.search('turquoise marker',search_mode='content')
        self.assertEqual([h.path for h in hits],[str(self.long.resolve())])
        self.assertIn('turquoise marker',hits[0].snippet)
        self.engine._model.encode.assert_not_called()

    def test_missing_filename_is_not_a_semantic_guess(self):
        self.assertEqual(self.engine.search('no_such_filename_742.txt',search_mode='filename'),[])
        self.assertEqual(self.engine.search('no such literal passage 742',search_mode='content'),[])
        self.engine._model.encode.assert_not_called()

    def test_scope_applies_to_literal_modes(self):
        for mode,query in [('filename','Report'),('content','delivery details')]:
            hits=self.engine.search(query,search_mode=mode,path_filter=[str(self.root/'beta')])
            self.assertEqual(len(hits),1)
            self.assertEqual(Path(hits[0].path).parent,(self.root/'beta').resolve())
            self.assertEqual(self.engine.search(query,search_mode=mode,path_filter=[]),[])

    def test_missing_and_changed_files_are_not_presented_as_current_results(self):
        first=self.root/'alpha'/'Report June.txt'
        first.unlink()
        other=self.root/'beta'/'Report June.txt'
        other.write_text('A different replacement without the indexed information.',encoding='utf-8')
        for mode,query in [('filename','Report'),('content','delivery details')]:
            self.assertEqual(self.engine.search(query,search_mode=mode),[])
        self.engine.index_directory(self.root/'beta')
        self.assertEqual(len(self.engine.search('Report',search_mode='filename')),1)

    def test_empty_scope_and_empty_index_skip_model_loading(self):
        self.assertEqual(self.engine.search('review',path_filter=[]),[])
        empty=FileSearchEngine(self.root/'empty.db')
        empty._model=Mock()
        self.assertEqual(empty.search('review'),[])
        empty._model.encode.assert_not_called()
        self.engine._model.encode.assert_not_called()

    def test_invalid_mode_and_zero_limit(self):
        with self.assertRaises(ValueError):
            self.engine.search('report',search_mode='unknown')
        self.assertEqual(self.engine.search('report',top_k=0),[])

    def test_empty_and_short_documents_are_findable_by_name(self):
        for name,text in [('empty.txt',''),('tiny.txt','OK')]:
            (self.root/'alpha'/name).write_text(text,encoding='utf-8')
        self.engine.index_directory(self.root/'alpha')
        self.assertEqual(len(self.engine.search('empty.txt',search_mode='filename')),1)
        self.assertEqual(len(self.engine.search('tiny.txt',search_mode='filename')),1)

    def test_literal_query_does_not_require_healthy_embeddings(self):
        with self.engine._conn() as conn:
            conn.execute('UPDATE files SET embedding=?',(b'broken',))
        self.assertEqual(len(self.engine.search('Report',search_mode='filename')),2)
        with self.assertRaises(RuntimeError):
            self.engine.search('review')


class SearchModeUITests(unittest.TestCase):
    def test_selected_duplicate_uses_its_exact_folder_and_missing_file_is_blocked(self):
        from ui.file_search_view import FileSearchPanel
        from PyQt6.QtWidgets import QListWidgetItem
        from PyQt6.QtCore import Qt
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('one','two'):
                (root/name).mkdir()
                (root/name/'same.txt').write_text('content',encoding='utf-8')
            panel=FileSearchPanel(str(root/'index.db'),[str(root)])
            item=QListWidgetItem()
            target=root/'two'/'same.txt'
            item.setData(Qt.ItemDataRole.UserRole,str(target))
            with patch('ui.file_search_view.QDesktopServices.openUrl',return_value=True) as opening, patch('ui.file_search_view.QMessageBox.warning') as warning:
                panel._on_result_double_clicked(item)
                self.assertEqual(Path(opening.call_args.args[0].toLocalFile()),target.parent)
                target.unlink()
                opening.reset_mock()
                panel._on_result_double_clicked(item)
                opening.assert_not_called()
                warning.assert_called_once()
            panel.deleteLater()

    def test_open_action_rechecks_changed_authorized_scope(self):
        from ui.file_search_view import FileSearchPanel
        from PyQt6.QtWidgets import QListWidgetItem
        from PyQt6.QtCore import Qt
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            target=root/'outside.txt'
            target.write_text('content',encoding='utf-8')
            panel=FileSearchPanel(str(root/'index.db'),[])
            item=QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole,str(target))
            with patch('ui.file_search_view.QDesktopServices.openUrl') as opening, patch('ui.file_search_view.QMessageBox.warning') as warning:
                panel._on_result_double_clicked(item)
                opening.assert_not_called()
                warning.assert_called_once()
            panel.deleteLater()

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def test_worker_passes_mode_without_opening_files(self):
        from ui.file_search_view import SearchWorker
        engine=Mock()
        engine.search.return_value=[]
        worker=SearchWorker(engine,'report',path_filter=['folder'],search_mode='filename')
        worker.run()
        engine.search.assert_called_once_with('report',path_filter=['folder'],status_filter=None,search_mode='filename')

    def test_literal_result_is_not_labelled_as_similarity(self):
        from core.file_intel import FileResult
        from ui.file_search_view import ResultItem
        from PyQt6.QtWidgets import QLabel
        result=FileResult(path='sample.txt',score=.9,snippet='text',file_type='document',fields={'search_mode':'filename'},modified_at='')
        widget=ResultItem(result)
        self.assertIn('Literal match',[w.text() for w in widget.findChildren(QLabel)])
        widget.deleteLater()


if __name__=='__main__':
    unittest.main()
