import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.interaction_policy import InteractionPolicy
from core.router import route, ExecutionMode
from core.window_handoff import DesktopHandoff, Window
from core.document_evidence import DocumentEvidence
from core.file_destination import desktop_output
from core.plugins import PluginRegistry
from core.skills import SkillLibrary


def action(kind, text=''):
    return SimpleNamespace(action_type=kind, text=text)


class GeneralEditorTests(unittest.TestCase):
    def test_default_allows_shortcuts_and_typing(self):
        policy = InteractionPolicy('Open VS Code and create report.py')
        for item in (action('key', 'ctrl+n'), action('type', 'user content')):
            self.assertEqual(policy.restriction(item), '')

    def test_mouse_launch_releases_keyboard_only_after_verified_launch(self):
        policy = InteractionPolicy('Open VS Code using mouse interaction, then write code')
        self.assertEqual(policy.scope, 'launch')
        self.assertTrue(policy.restriction(action('key', 'win')))
        self.assertTrue(policy.restriction(action('type', 'VS Code')))
        policy.launch_complete = True
        self.assertEqual(policy.restriction(action('key', 'ctrl+n')), '')
        self.assertEqual(policy.restriction(action('type', 'requested content')), '')

    def test_whole_task_mouse_constraint_never_releases_typing(self):
        policy = InteractionPolicy('Open VS Code using only the mouse for the whole task, then create a file')
        policy.launch_complete = True
        self.assertTrue(policy.restriction(action('type', 'code')))
        self.assertEqual(policy.restriction(action('click')), '')

    def test_save_scope_does_not_block_normal_editing(self):
        policy = InteractionPolicy('Write code then save using mouse interaction')
        self.assertEqual(policy.scope, 'save')
        self.assertEqual(policy.restriction(action('key', 'enter')), '')
        self.assertTrue(policy.restriction(action('key', 'ctrl+s')))
        self.assertTrue(policy.restriction(action('key', 'enter'), SimpleNamespace(foreground_window_title='Save As')))

    def test_code_url_does_not_route_editor_task_to_browser(self):
        self.assertEqual(route('Open VS Code and create link.py containing https://example.org').mode, ExecutionMode.DESKTOP)
        self.assertEqual(route('Create link.py on Desktop containing https://example.org').mode, ExecutionMode.DESKTOP)
        self.assertEqual(route('Visit https://example.org').mode, ExecutionMode.WEB)

    def test_existing_vscode_is_reused(self):
        window = Window(7, 'Untitled - Visual Studio Code', 'Code.exe')
        backend = Mock()
        backend.foreground.return_value = window
        backend.windows.return_value = [window]
        result = DesktopHandoff(backend)('Open VS Code and create a Python file')
        self.assertTrue(result.success)
        self.assertTrue(result.target_verified)
        self.assertFalse(result.complete)
        backend.launch.assert_not_called()

    def test_closed_mouse_launch_exposes_launcher_without_executable(self):
        backend = Mock()
        backend.windows.return_value = []
        backend.prepare_launcher.return_value = True
        result = DesktopHandoff(backend)('Open VS Code using mouse, then create a file')
        self.assertTrue(result.success)
        self.assertFalse(result.target_verified)
        self.assertFalse(result.complete)
        backend.launch.assert_not_called()

    @patch('core.file_destination.desktop_path', return_value=Path('C:/Users/Test/OneDrive/Desktop'))
    def test_redirected_desktop_and_ambiguous_names(self, desktop):
        self.assertEqual(desktop_output('Create report.py on Desktop'), str(desktop.return_value / 'report.py'))
        self.assertIsNone(desktop_output('Create a.py and b.py on Desktop'))
        self.assertIsNone(desktop_output('Read report.py on Desktop'))
        self.assertIn(str(desktop.return_value), SkillLibrary().context('Create report.py on Desktop'))

    @patch('core.file_destination.desktop_path', return_value=Path('C:/Test/Desktop'))
    @patch('core.document_evidence._read')
    def test_code_file_requires_disk_editor_match(self, read, desktop):
        evidence = DocumentEvidence('Create report.py on Desktop')
        editor = SimpleNamespace(role='EditControl', name='Editor content', text='print(123)\n')
        state = SimpleNamespace(elements=[editor], foreground_window_title='report.py - Visual Studio Code')
        read.return_value = b'print(123)\n'
        self.assertEqual(evidence.check(state), '')
        read.return_value = b'wrong content'
        self.assertIn('do not match', evidence.check(state))
        read.side_effect = FileNotFoundError()
        self.assertIn('missing or unreadable', evidence.check(state))

    def test_vscode_workflow_is_parameterized_and_window_scoped(self):
        registry = PluginRegistry(Path(__file__).resolve().parents[1] / 'data/plugins')
        registry.discover()
        match = registry.find_workflow('create analysis.py', 'Untitled - Visual Studio Code')
        self.assertIsNotNone(match)
        self.assertEqual(match[1].name, 'create_code_file')
        self.assertEqual({s.parameter for s in match[1].steps if s.parameter}, {'requested_content', 'output_full_path'})
        self.assertIsNone(registry.find_workflow('create analysis.py', 'Calculator'))


if __name__ == '__main__':
    unittest.main()
