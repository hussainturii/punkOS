import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core.plugins import PluginRegistry
from core.atomic_storage import write_json


class PluginLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = PluginRegistry(self.root / 'plugins')
        self.manifest = json.loads((Path(__file__).resolve().parents[1] / 'data/plugins/notepad/plugin.json').read_text())
        self.assertTrue(self.registry.install_from_dict(self.manifest)[0])

    def test_failed_install_preserves_previous_manifest(self):
        path = self.root / 'plugins/notepad/plugin.json'
        previous = path.read_bytes()
        changed = {**self.manifest, 'version': '9.0.0'}
        with patch('core.plugins.os.replace', side_effect=PermissionError('locked')):
            self.assertFalse(self.registry.install_from_dict(changed, overwrite=True)[0])
        self.assertEqual(path.read_bytes(), previous)
        self.assertNotEqual(self.registry.plugins['notepad'].version, '9.0.0')

    def test_uninstall_rejects_outside_and_root_paths(self):
        for target in (self.root, self.registry.directory):
            self.registry.plugins['notepad'].directory = str(target)
            self.assertFalse(self.registry.uninstall('notepad')[0])
            self.assertTrue(target.exists())

    def test_uninstall_removes_only_owned_plugin(self):
        keep = self.root / 'keep.txt'
        keep.write_text('preserve')
        self.assertTrue(self.registry.uninstall('notepad')[0])
        self.assertFalse((self.root / 'plugins/notepad').exists())
        self.assertEqual(keep.read_text(), 'preserve')

    def test_knowledge_collision_is_preserved(self):
        source = self.root / 'guide.txt'
        source.write_text('first')
        self.assertTrue(self.registry.add_knowledge_file('notepad', str(source))[0])
        source.write_text('second')
        self.assertFalse(self.registry.add_knowledge_file('notepad', str(source))[0])
        self.assertEqual((self.root / 'plugins/notepad/knowledge/guide.txt').read_text(), 'first')

    def test_knowledge_persistence_failure_is_not_success(self):
        source = self.root / 'guide.txt'
        source.write_text('content')
        with patch.object(self.registry, '_persist', return_value=False):
            ok, message = self.registry.add_knowledge_file('notepad', str(source))
        self.assertFalse(ok)
        self.assertIn('unregistered', message)
        self.assertNotIn('guide.txt', self.registry.plugins['notepad'].knowledge_base)

    def test_atomic_settings_failure_preserves_valid_previous_json(self):
        path = self.root / 'config.json'
        write_json(path, {'model': 'previous'})
        with patch('core.atomic_storage.os.replace', side_effect=PermissionError('locked')):
            with self.assertRaises(PermissionError):
                write_json(path, {'model': 'new'})
        self.assertEqual(json.loads(path.read_text()), {'model': 'previous'})
        self.assertEqual(list(self.root.glob('*.tmp')), [])
