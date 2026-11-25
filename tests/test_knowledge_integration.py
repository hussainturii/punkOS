import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
import json
from core.plugins import PluginRegistry
from PyQt6.QtWidgets import QApplication
from ui.plugin_manager import PluginManagerPanel


class KnowledgeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.registry = PluginRegistry(self.root/'plugins')
        manifest = json.loads((Path(__file__).resolve().parents[1]/'data/plugins/notepad/plugin.json').read_text())
        self.registry.install_from_dict(manifest)
        self.source = self.root/'guide.txt'
        self.source.write_text('Use File then Save to preserve a note.')
        self.registry.add_knowledge_file('notepad', str(self.source))
        self.registered = self.root/'plugins/notepad/knowledge/guide.txt'
        self.engine = Mock()
        self.engine.search.return_value = [SimpleNamespace(path=str(self.registered), snippet='Use File then Save.')]
        self.engine.index_files.return_value = 1
        self.registry.knowledge_engine = self.engine

    def test_scoped_reference_enters_hint_without_repeating_query(self):
        first = self.registry.hint_for_prompt('Untitled - Notepad', 'write and save')
        second = self.registry.hint_for_prompt('Untitled - Notepad', 'write and save')
        self.assertEqual(first, second)
        self.assertIn('untrusted data', first)
        self.assertIn(json.dumps(str(self.registered)), first)
        self.engine.search.assert_called_once()
        self.assertEqual(self.engine.search.call_args.kwargs['path_filter'], [str(self.registered)])

    def test_unregistered_hits_and_disabled_plugin_do_not_enter_prompt(self):
        self.engine.search.return_value = [SimpleNamespace(path=str(self.source), snippet='outside')]
        self.assertNotIn('outside', self.registry.knowledge_for_prompt('Untitled - Notepad', 'save'))
        self.registry.plugins['notepad'].is_active = False
        self.assertEqual(self.registry.hint_for_prompt('Untitled - Notepad', 'save'), '')

    def test_changed_document_invalidates_cached_excerpt(self):
        self.registry.knowledge_for_prompt('Untitled - Notepad', 'save')
        self.registered.write_text('Different content and size')
        self.engine.search.return_value = []
        self.assertEqual(self.registry.knowledge_for_prompt('Untitled - Notepad', 'save'), '')
        self.assertEqual(self.engine.search.call_count, 2)

    def test_dashboard_indexes_on_worker_and_reports_result(self):
        import threading
        threads = []
        def index(paths):
            threads.append(threading.get_ident())
            return 1
        self.engine.index_files.side_effect = index
        panel = PluginManagerPanel(self.registry)
        panel.current_plugin_id = 'notepad'
        try:
            panel._on_index_knowledge()
            self.assertFalse(panel.index_btn.isEnabled())
            self.assertTrue(panel.shutdown(5000))
            self.app.processEvents()
            self.assertIn('Indexed 1', panel.kb_hint.text())
            self.assertNotEqual(threads[0], threading.get_ident())
        finally:
            panel.shutdown(5000)
            panel.close()
