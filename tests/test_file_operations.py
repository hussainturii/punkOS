import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from core.file_operations import run_file_operation
from core.llm import LLMResponse
from core.safety import SafetySystem
from core.router import route, ExecutionMode


class FileOperationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / 'source.txt'
        self.source.write_bytes(b'example\x00data\n')
        self.dest = self.root / 'destination.txt'
        self.safety = SafetySystem([str(self.root)])
        self.provider = Mock()

    def run_op(self, kind='copy', stopped=lambda: False, goal=None):
        self.provider.call_with_schema.return_value = LLMResponse({
            'operation': kind, 'source': '' if kind == 'mkdir' else str(self.source),
            'destination': str(self.dest), 'reason': ''}, '')
        goal = goal or (f'Create folder "{self.dest}".' if kind == 'mkdir' else
                        f'{kind.title()} file "{self.source}" to "{self.dest}". Do not overwrite anything.')
        return run_file_operation(goal, self.provider, self.safety, stopped)

    def test_copy_verifies_bytes_and_preserves_source(self):
        before = self.source.read_bytes()
        success, message, history = self.run_op()
        self.assertTrue(success, message)
        self.assertEqual(self.dest.read_bytes(), before)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertIn('SHA-256', message)
        self.assertEqual(len(history), 1)

    def test_cancel_during_copy_reports_partial_destination(self):
        calls = [0]
        def stopped():
            calls[0] += 1
            return calls[0] >= 4
        success, message, _ = self.run_op(stopped=stopped)
        self.assertFalse(success)
        self.assertIn('partial', message)
        self.assertIn(str(self.dest), message)
        self.assertTrue(self.source.exists())

    def test_destination_collision_after_preflight_is_not_overwritten(self):
        original_open = Path.open
        def racing_open(path, mode='r', *args, **kwargs):
            if path == self.dest and mode == 'xb':
                with original_open(path, 'wb') as handle:
                    handle.write(b'concurrent writer')
            return original_open(path, mode, *args, **kwargs)
        with patch.object(Path, 'open', racing_open):
            success, _, _ = self.run_op()
        self.assertFalse(success)
        self.assertEqual(self.dest.read_bytes(), b'concurrent writer')

    def test_corrupt_destination_is_never_reported_as_success(self):
        from core.file_operations import _digest
        def digest(path):
            return 'different' if path == self.dest else _digest(path)
        with patch('core.file_operations._digest', side_effect=digest):
            success, message, _ = self.run_op()
        self.assertFalse(success)
        self.assertIn('verification failed', message)
        self.assertIn('may remain', message)

    def test_collision_never_overwrites(self):
        self.dest.write_text('keep')
        self.assertFalse(self.run_op()[0])
        self.assertEqual(self.dest.read_text(), 'keep')

    def test_create_folder(self):
        self.assertTrue(self.run_op('mkdir')[0])
        self.assertTrue(self.dest.is_dir())

    @unittest.skipUnless(os.name == 'nt', 'Windows no-overwrite rename')
    def test_move_and_rename_preserve_bytes(self):
        original = self.source.read_bytes()
        self.assertTrue(self.run_op('rename')[0])
        self.assertFalse(self.source.exists())
        self.assertEqual(self.dest.read_bytes(), original)
        self.source, self.dest = self.dest, self.root / 'moved.txt'
        self.assertTrue(self.run_op('move')[0])
        self.assertEqual(self.dest.read_bytes(), original)

    def test_missing_source_and_missing_parent(self):
        self.source.unlink()
        self.assertFalse(self.run_op()[0])
        self.assertFalse(self.dest.exists())
        self.source.write_text('data')
        self.dest = self.root / 'missing' / 'output.txt'
        self.assertFalse(self.run_op()[0])
        self.assertFalse(self.dest.parent.exists())

    def test_blocked_path_denied_before_model(self):
        self.safety = SafetySystem([str(self.root)], blocked_paths=[str(self.source)])
        self.assertFalse(self.run_op()[0])
        self.provider.call_with_schema.assert_not_called()

    def test_cancelled_before_planning(self):
        self.assertFalse(self.run_op(stopped=lambda: True)[0])
        self.provider.call_with_schema.assert_not_called()
        self.assertFalse(self.dest.exists())

    def test_preview_cannot_execute_even_if_model_would_plan_copy(self):
        self.assertFalse(self.run_op(goal=f'Copy "{self.source}" to "{self.dest}" as a dry run.')[0])
        self.provider.call_with_schema.assert_not_called()
        self.assertFalse(self.dest.exists())

    def test_model_cannot_change_copy_into_move(self):
        self.assertFalse(self.run_op('move', goal=f'Copy "{self.source}" to "{self.dest}".')[0])
        self.assertTrue(self.source.exists())

    def test_model_invented_destination_rejected(self):
        goal = f'Copy "{self.source}" to "{self.root / "requested.txt"}".'
        self.assertFalse(self.run_op(goal=goal)[0])
        self.assertFalse(self.dest.exists())

    def test_routing_respects_explicit_ui_and_editor_workflows(self):
        goal = f'Copy file "{self.source}" to "{self.dest}".'
        self.assertEqual(route(goal).mode, ExecutionMode.FILE_OPERATION)
        self.assertEqual(route(goal + ' Use mouse interaction.').mode, ExecutionMode.DESKTOP)
        self.assertEqual(route('Open VS Code and create a file.').mode, ExecutionMode.DESKTOP)

    def test_planner_integration_does_not_use_desktop_or_learning(self):
        from core.planner import StrategicPlanner
        perception, reasoning, controller, handoff, learner = [Mock() for _ in range(5)]
        self.run_op()  # configure response, remove the fixture destination
        self.dest.unlink()
        reasoning.provider = self.provider
        planner = StrategicPlanner(perception, reasoning, controller, self.safety,
                                   desktop_handoff=handoff, learner=learner, on_log=lambda _: None)
        result = planner.run(f'Copy file "{self.source}" to "{self.dest}".')
        self.assertTrue(result.success, result.final_message)
        self.assertEqual(result.execution_mode, 'FILE_OPERATION')
        perception.perceive.assert_not_called()
        controller.execute.assert_not_called()
        handoff.assert_not_called()
        learner.learn_from_task.assert_not_called()
