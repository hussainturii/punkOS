import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from core.file_destination import existing_file_stop
from core.planner import StrategicPlanner
from core.safety import SafetySystem


class ExistingFilePreflightTests(unittest.TestCase):
    separate_goal = ('Open VS Code and open punkos_general_b.py from my Desktop. Keep its contents unchanged. '
        'Create a separate new file containing exactly print("Separate document test") and save it on my Desktop as punkos_general_c.py. '
        'If that destination already exists, stop without overwriting it. Do not run either file. '
        'Verify the new file’s contents and confirm the original file remains unchanged.')

    def test_existing_source_is_not_destination(self):
        from core.file_destination import desktop_output
        from core.document_evidence import DocumentEvidence
        with tempfile.TemporaryDirectory() as directory, patch('core.file_destination.desktop_path', return_value=Path(directory)):
            source = Path(directory)/'punkos_general_b.py'
            output = Path(directory)/'punkos_general_c.py'
            source.write_bytes(b'original')
            self.assertEqual(desktop_output(self.separate_goal), str(output))
            self.assertIsNone(existing_file_stop(self.separate_goal))
            evidence = DocumentEvidence(self.separate_goal)
            self.assertEqual(evidence.source, str(source))
            self.assertEqual(evidence.output, str(output))
            self.assertEqual(evidence.source_prerequisite(Mock(), []), '')
            source.write_bytes(b'changed')
            self.assertIn('protected source changed', evidence.check(Mock()))
            output.write_bytes(b'existing output')
            result = existing_file_stop(self.separate_goal)
            self.assertTrue(result[0])
            self.assertIn(str(output), result[1])
            self.assertNotIn(str(source), result[1])

    def test_trailing_punctuation_does_not_hide_destination(self):
        from core.file_destination import desktop_output
        with patch('core.file_destination.desktop_path', return_value=Path('Desktop')):
            for ending in ('.', ',', ';', '!', '?', ''):
                self.assertEqual(desktop_output('Create result.py' + ending + ' on Desktop'), str(Path('Desktop')/'result.py') if ending not in {'.','!','?'} else None)
            self.assertEqual(desktop_output('Save it on Desktop as result.py.'),str(Path('Desktop')/'result.py'))

    goal = 'Create example.py on my Desktop containing print("new"). If that filename already exists, stop without overwriting.'

    def test_existing_file_is_success_without_model_or_app_actions(self):
        with tempfile.TemporaryDirectory() as directory, patch('core.file_destination.desktop_path', return_value=Path(directory)):
            target = Path(directory)/'example.py'
            target.write_bytes(b'original')
            perception, reasoning, controller, handoff = Mock(), Mock(), Mock(), Mock()
            planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
                desktop_handoff=handoff, on_log=lambda _:None)
            result = planner.run(self.goal)
            self.assertTrue(result.success)
            self.assertIn(str(target), result.final_message)
            self.assertEqual(target.read_bytes(), b'original')
            perception.perceive.assert_not_called()
            reasoning.plan_task.assert_not_called()
            reasoning.decide_next_action.assert_not_called()
            reasoning.provider.call_with_schema.assert_not_called()
            controller.execute.assert_not_called()
            handoff.assert_not_called()

    def test_absent_target_continues_and_no_file_is_created(self):
        with tempfile.TemporaryDirectory() as directory, patch('core.file_destination.desktop_path', return_value=Path(directory)):
            self.assertIsNone(existing_file_stop(self.goal))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_ambiguous_destination_does_not_guess(self):
        result = existing_file_stop('Create a.py and b.py on Desktop. If the file exists, stop.')
        self.assertFalse(result[0])

    def test_unreadable_destination_does_not_grant_permission(self):
        with patch('pathlib.Path.lstat', side_effect=PermissionError()):
            result = existing_file_stop(self.goal)
            self.assertFalse(result[0])
            self.assertIn('Could not check', result[1])

    def test_unconditional_create_and_literal_content_are_not_stop_requests(self):
        self.assertIsNone(existing_file_stop('Create example.py on Desktop containing "if it exists stop"'))
        self.assertIsNone(existing_file_stop('Create example.py on Desktop and replace the existing file.'))

    def test_full_path_and_filename_in_condition(self):
        with patch('pathlib.Path.lstat', return_value=Mock()):
            result = existing_file_stop(r'Create C:\Test\example.py. If example.py exists, stop.')
            self.assertTrue(result[0])
            self.assertIn(r'C:\Test\example.py', result[1])

    def test_absolute_source_cannot_be_used_as_ambiguous_destination(self):
        result = existing_file_stop(r'Open C:\Test\source.py. Create another file. If that destination exists, stop.')
        self.assertFalse(result[0])
        self.assertIn('Cannot resolve', result[1])


if __name__ == '__main__':
    unittest.main()
