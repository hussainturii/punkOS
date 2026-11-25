import unittest
from unittest.mock import Mock, patch
from tests import test_plugin_lifecycle
from core.plugins import validate_manifest
from core.file_intel import FileSearchEngine, IndexingError


class KnowledgeScopeTests(unittest.TestCase):
    setUp = test_plugin_lifecycle.PluginLifecycleTests.setUp
    def register(self):
        source = self.root / 'registered.txt'
        source.write_text('reference')
        self.assertTrue(self.registry.add_knowledge_file('notepad', str(source))[0])
        return self.root / 'plugins/notepad/knowledge'

    def test_unregistered_files_and_subfolders_are_not_indexed(self):
        folder = self.register()
        (folder / 'orphan.txt').write_text('unregistered')
        (folder / 'nested').mkdir()
        (folder / 'nested/other.txt').write_text('outside list')
        engine = Mock()
        engine.index_files.return_value = 1
        self.assertTrue(self.registry.index_knowledge_base('notepad', engine)[0])
        self.assertEqual(engine.index_files.call_args.args[0], [(folder / 'registered.txt').resolve()])
        engine.index_directory.assert_not_called()

    def test_missing_registration_fails_before_any_indexing(self):
        self.register()
        self.registry.plugins['notepad'].knowledge_base.append('missing.txt')
        engine = Mock()
        self.assertFalse(self.registry.index_knowledge_base('notepad', engine)[0])
        engine.index_files.assert_not_called()

    def test_disabled_plugin_does_not_index(self):
        self.register()
        self.registry.plugins['notepad'].is_active = False
        engine = Mock()
        self.assertFalse(self.registry.index_knowledge_base('notepad', engine)[0])
        engine.index_files.assert_not_called()

    def test_partial_copy_is_removed_without_touching_source(self):
        source = self.root / 'copy.txt'
        source.write_text('preserve')
        def fail(reader, writer):
            writer.write(b'partial')
            raise OSError('disk failure')
        with patch('core.plugins.shutil.copyfileobj', side_effect=fail):
            self.assertFalse(self.registry.add_knowledge_file('notepad', str(source))[0])
        self.assertFalse((self.root / 'plugins/notepad/knowledge/copy.txt').exists())
        self.assertEqual(source.read_text(), 'preserve')

    def test_manifest_rejects_nested_and_stream_paths(self):
        for name in ('nested/file.txt', 'nested\\file.txt', 'file.txt:stream'):
            self.assertTrue(validate_manifest({**self.manifest, 'knowledge_base': [name]}))

    def test_explicit_index_list_reports_partial_failure(self):
        folder = self.register()
        engine = FileSearchEngine(self.root / 'index.db')
        path = folder / 'registered.txt'
        with patch.object(engine, '_index_one', return_value=True) as index:
            with self.assertRaises(IndexingError) as error:
                engine.index_files([path, path, folder / 'missing.txt'])
        self.assertEqual(error.exception.indexed, 1)
        index.assert_called_once()
