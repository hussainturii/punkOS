import unittest
from unittest.mock import Mock, patch

from core.window_handoff import DesktopHandoff, HandoffResult, Window, launch_only


class HandoffTests(unittest.TestCase):
    def test_enumeration_excludes_owned_auxiliary_windows(self):
        from core.window_handoff import WindowsBackend
        backend = WindowsBackend.__new__(WindowsBackend)
        backend.user = Mock()
        backend.callback_type = lambda callback: callback
        backend.user.IsWindowVisible.side_effect = lambda handle: handle != 4
        backend.user.GetWindow.side_effect = lambda handle, flag: 1 if handle == 2 else 0
        def enumerate_windows(callback, _):
            for handle in (1, 2, 3, 4):
                callback(handle, 0)
            return True
        backend.user.EnumWindows.side_effect = enumerate_windows
        backend._window = Mock(side_effect=lambda handle: Window(handle, 'Note - Notepad', 'notepad.exe'))
        self.assertEqual([w.handle for w in backend.windows()], [1, 3])
        self.assertEqual(backend._window.call_count, 2)

    def test_hosted_identity_requires_supported_child_process(self):
        from core.window_handoff import hosted_executable
        self.assertEqual(hosted_executable('ApplicationFrameHost.exe', ['Calculator.exe']), 'calculator.exe')
        self.assertEqual(hosted_executable('ApplicationFrameHost.exe', ['unrelated.exe']), 'ApplicationFrameHost.exe')
        self.assertEqual(hosted_executable('ApplicationFrameHost.exe', ['Calculator.exe', 'notepad.exe']), 'ApplicationFrameHost.exe')
        self.assertEqual(hosted_executable('unrelated.exe', ['Calculator.exe']), 'unrelated.exe')

    @patch('core.window_handoff.time.sleep')
    def test_delayed_foreground_activation_is_verified(self, sleep):
        self.backend.foreground.side_effect = [self.punkos, self.punkos, self.punkos, self.calc]
        result = self.handoff('Open Calculator')
        self.assertTrue(result.success)
        self.backend.launch.assert_not_called()
        self.assertEqual(sleep.call_count, 2)

    def test_closed_notepad_is_launched_and_verified(self):
        note = Window(9, 'Untitled - Notepad', 'notepad.exe')
        self.backend.windows.side_effect = [[], [note]]
        self.backend.foreground.side_effect = [self.punkos, note]
        result = self.handoff('open notepad')
        self.assertTrue(result.success)
        self.assertTrue(result.complete)
        self.backend.launch.assert_called_once_with('notepad')

    def test_open_existing_app_does_not_spawn_duplicate(self):
        self.backend.foreground.side_effect = [self.punkos, self.calc]
        self.assertTrue(self.handoff('Please open Calculator').complete)
        self.backend.launch.assert_not_called()

    @patch('core.window_handoff.time.sleep')
    def test_launch_without_window_does_not_claim_success(self, sleep):
        self.backend.windows.return_value = []
        result = self.handoff('open notepad')
        self.assertFalse(result.success)
        self.assertFalse(result.complete)
        self.backend.launch.assert_called_once_with('notepad')

    def test_launch_with_more_work_is_not_completion(self):
        self.backend.foreground.side_effect = [self.punkos, self.calc]
        result = self.handoff('Open Calculator and calculate 27 times 14')
        self.assertTrue(result.success)
        self.assertFalse(result.complete)

    def test_negated_launch_and_quoted_instruction_do_not_launch(self):
        self.backend.windows.return_value = []
        for goal in ("Do not open Notepad", "Type 'open Notepad' into the editor"):
            self.assertFalse(self.handoff(goal).success)
        self.backend.launch.assert_not_called()

    def test_closed_mouse_only_app_does_not_launch(self):
        self.backend.prepare_launcher.return_value = False
        self.backend.windows.return_value = []
        self.assertFalse(self.handoff('Open Notepad using mouse interaction').success)
        self.backend.launch.assert_not_called()

    def test_only_exact_launch_goals_are_complete(self):
        for goal in ('open notepad', 'Please open Notepad.', 'Can you launch the Notepad app?'):
            self.assertTrue(launch_only(goal, 'notepad'))
        for goal in ('Open Notepad and type Hello', 'Open Notepad then save a file', 'Do not open Notepad'):
            self.assertFalse(launch_only(goal, 'notepad'))

    def test_verified_launch_only_skips_model_and_records_success(self):
        from core.planner import StrategicPlanner
        perception, reasoning, controller, safety, history = [Mock() for _ in range(5)]
        planner = StrategicPlanner(perception, reasoning, controller, safety,
            history_store=history, desktop_handoff=lambda _: HandoffResult(True, 'Notepad foreground verified', True), on_log=lambda _: None)
        result = planner.run('Open Notepad')
        self.assertTrue(result.success)
        self.assertEqual(result.steps_taken, 1)
        reasoning.plan_task.assert_not_called()
        reasoning.decide_next_action.assert_not_called()
        perception.perceive.assert_not_called()
        history.add_task.assert_called_once()

    def setUp(self):
        self.punkos = Window(1, 'PunkOS — Agentic Desktop AI', 'pythonw.exe')
        self.calc = Window(2, 'Calculator', 'CalculatorApp.exe')
        self.backend = Mock()
        self.backend.foreground.return_value = self.punkos
        self.backend.windows.return_value = [self.punkos, self.calc]
        self.backend.click_taskbar.return_value = False
        self.handoff = DesktopHandoff(self.backend)

    def test_focus_is_verified_after_activation(self):
        self.backend.foreground.side_effect = [self.punkos, self.calc]
        self.assertTrue(self.handoff('Use Calculator to calculate 27 times 14').success)
        self.backend.activate.assert_called_once_with(2)

    def test_already_focused_does_not_activate(self):
        self.backend.foreground.return_value = self.calc
        self.assertTrue(self.handoff('Use Calculator').success)
        self.backend.activate.assert_not_called()

    def test_os_denial_fails(self):
        self.assertFalse(self.handoff('Use Calculator').success)

    def test_title_spoof_is_not_selected(self):
        self.backend.windows.return_value = [Window(3, 'Calculator', 'chrome.exe')]
        self.assertFalse(self.handoff('Use Calculator').success)
        self.backend.activate.assert_not_called()

    def test_ambiguous_windows_fail(self):
        self.backend.windows.return_value = [self.calc, Window(4, 'Calculator', 'CalculatorApp.exe')]
        self.assertFalse(self.handoff('Use Calculator').success)
        self.backend.activate.assert_not_called()

    def test_multiple_apps_fail(self):
        self.assertFalse(self.handoff('Read Notepad and use Calculator').success)
        self.backend.activate.assert_not_called()

    def test_mouse_only_does_not_use_native_activation(self):
        self.assertFalse(self.handoff('Open Calculator using mouse interaction').success)
        self.backend.activate.assert_not_called()

    def test_mouse_only_taskbar_click_requires_exact_foreground(self):
        self.backend.click_taskbar.return_value = True
        self.backend.foreground.side_effect = [self.punkos, self.calc]
        self.assertTrue(self.handoff('Use Calculator with mouse only').success)
        self.backend.click_taskbar.assert_called_once_with('calculator')
        self.backend.activate.assert_not_called()
        self.backend.launch.assert_not_called()

    @patch('core.window_handoff.time.sleep')
    def test_mouse_click_that_opens_preview_does_not_claim_focus(self, sleep):
        self.backend.click_taskbar.return_value = True
        self.assertFalse(self.handoff('Use Calculator with mouse only').success)
        self.backend.click_taskbar.assert_called_once()
        self.backend.activate.assert_not_called()

    def test_unknown_task_does_not_guess_other_windows(self):
        self.assertFalse(self.handoff('Calculate this result').success)
        self.backend.activate.assert_not_called()

    def test_unknown_task_can_use_existing_external_foreground(self):
        self.backend.foreground.return_value = Window(8, 'Editor', 'editor.exe')
        self.assertTrue(self.handoff('Type a note').success)
        self.backend.activate.assert_not_called()

    def test_backend_error_fails_closed(self):
        self.backend.windows.side_effect = OSError('unavailable')
        self.assertFalse(self.handoff('Use Calculator').success)

    def test_failed_handoff_never_calls_perception_or_model(self):
        from core.planner import StrategicPlanner
        perception, reasoning, controller, safety, history = [Mock() for _ in range(5)]
        planner = StrategicPlanner(perception, reasoning, controller, safety,
            history_store=history, desktop_handoff=lambda _: HandoffResult(False, 'No target'), on_log=lambda _: None)
        result = planner.run('Use Calculator')
        self.assertFalse(result.success)
        self.assertEqual(result.steps_taken, 0)
        perception.perceive.assert_not_called()
        reasoning.plan_task.assert_not_called()
        reasoning.decide_next_action.assert_not_called()
        controller.execute.assert_not_called()
        history.add_task.assert_called_once()


if __name__ == '__main__':
    unittest.main()
