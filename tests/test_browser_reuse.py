import unittest
from unittest.mock import Mock, patch
from core.window_handoff import DesktopHandoff, Window


class BrowserReuseTests(unittest.TestCase):
    def test_open_page_reference_matches_observed_title_without_service_mapping(self):
        target = Window(7, 'Inbox - ExampleMail - Google Chrome', 'chrome.exe')
        self.backend.windows.return_value = [target, self.edge]
        self.backend.foreground.side_effect = [Window(9, 'PunkOS', 'python.exe'), target]
        result = self.handoff.reuse_browser('Inspect the currently open ExampleMail composer. Do not click.', match_open_title=True)
        self.assertTrue(result.success)
        self.backend.activate.assert_called_once_with(7)
        self.backend.launch.assert_not_called()

    def test_title_resolution_does_not_guess_when_reference_is_missing(self):
        self.assertIsNone(self.handoff.reuse_browser('Inspect the currently open UnknownService composer.', match_open_title=True))
        self.assertIsNone(self.handoff.reuse_browser('Read the current note', match_open_title=True))
        self.backend.activate.assert_not_called()

    def test_ambiguous_page_reference_does_not_pick_a_random_account(self):
        self.backend.windows.return_value = [Window(1, 'ExampleMail - account A', 'chrome.exe'),
                                            Window(2, 'ExampleMail - account B', 'chrome.exe')]
        self.backend.foreground.return_value = Window(9, 'PunkOS', 'python.exe')
        result = self.handoff.reuse_browser('Inspect the currently open ExampleMail composer.', match_open_title=True)
        self.assertFalse(result.success)
        self.backend.activate.assert_not_called()

    def test_inspection_prompt_dispatches_to_matching_page_without_input(self):
        from core.planner import StrategicPlanner
        from core.perception import ScreenState
        from core.reasoning import NextAction
        from core.safety import SafetySystem
        target = Window(7, 'ExampleMail - Chrome', 'chrome.exe')
        self.backend.windows.return_value = [target]
        self.backend.foreground.side_effect = [Window(9, 'PunkOS', 'python.exe'), target]
        perception, reasoning, controller = Mock(), Mock(), Mock()
        perception.perceive.return_value = ScreenState('', '', [], target.title, 100, 100)
        reasoning.decide_next_action.return_value = NextAction('fail', text='Value unreadable')
        from core.llm import LLMResponse
        reasoning.provider.call_with_schema.return_value = LLMResponse(
            dict(readable=False, findings=[], limitation='Value unreadable'), '')
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
            desktop_handoff=self.handoff, use_planning_phase=False, max_steps=1,
            step_delay=0, on_log=lambda _: None)
        result = planner.run('Inspect the currently open ExampleMail composer. Do not type or click.')
        self.assertIn('Value unreadable', result.final_message)
        reasoning.decide_next_action.assert_not_called()
        self.backend.activate.assert_called_once_with(7)
        controller.execute.assert_not_called()

    def setUp(self):
        self.chrome = Window(1, 'Inbox', 'chrome.exe')
        self.edge = Window(2, 'Page', 'msedge.exe')
        self.backend = Mock()
        self.backend.windows.return_value = [self.chrome]
        self.backend.foreground.return_value = self.chrome
        self.handoff = DesktopHandoff(self.backend)

    def test_existing_named_browser_is_reused_without_launch(self):
        self.assertTrue(self.handoff.reuse_browser('Open Chrome and go to Gmail').success)
        self.backend.launch.assert_not_called()
        self.backend.activate.assert_not_called()

    def test_foreground_browser_wins_among_multiple(self):
        self.backend.windows.return_value = [self.chrome, self.edge]
        self.assertTrue(self.handoff.reuse_browser('Visit https://example.test').success)
        self.backend.activate.assert_not_called()

    def test_missing_matching_browser_allows_guest_fallback(self):
        self.backend.windows.return_value = [self.edge]
        self.assertIsNone(self.handoff.reuse_browser('Open Chrome and visit Gmail'))
        self.backend.launch.assert_not_called()

    def test_no_browser_allows_guest_fallback(self):
        self.backend.windows.return_value = []
        self.assertIsNone(self.handoff.reuse_browser('Go to Gmail'))

    def test_ambiguity_and_enumeration_failure_do_not_allow_guest_fallback(self):
        self.backend.windows.return_value = [self.chrome, Window(3, 'Other account', 'chrome.exe')]
        self.backend.foreground.return_value = Window(9, 'PunkOS', 'python.exe')
        self.assertFalse(self.handoff.reuse_browser('Visit Gmail').success)
        self.backend.activate.assert_not_called()
        self.backend.windows.side_effect = RuntimeError('Unavailable')
        self.assertFalse(self.handoff.reuse_browser('Visit Gmail').success)

    def test_existing_background_window_is_focused_and_verified(self):
        self.backend.foreground.side_effect = [Window(9, 'PunkOS', 'python.exe'), self.chrome]
        self.assertTrue(self.handoff.reuse_browser('Visit Gmail').success)
        self.backend.activate.assert_called_once_with(1)

    @patch('core.window_handoff.time.sleep')
    def test_focus_failure_does_not_launch_guest(self, sleep):
        self.backend.foreground.return_value = Window(9, 'PunkOS', 'python.exe')
        self.assertFalse(self.handoff.reuse_browser('Visit Gmail').success)
        self.backend.launch.assert_not_called()

    def test_planner_does_not_launch_web_agent_when_existing_browser_is_available(self):
        from core.planner import StrategicPlanner
        from core.perception import ScreenState, UIElement
        from core.reasoning import NextAction
        from core.safety import SafetySystem
        perception, reasoning, controller, web = Mock(), Mock(), Mock(), Mock()
        perception.perceive.return_value = ScreenState('', '', [
            UIElement(1, 'ButtonControl', 'Compose', (0, 0, 30, 30), True)
        ], 'Inbox', 100, 100)
        reasoning.decide_next_action.return_value = NextAction('fail', text='Synthetic stop')
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
            web_agent=web, desktop_handoff=self.handoff, use_planning_phase=False,
            max_steps=1, step_delay=0, on_log=lambda _: None)
        result = planner.run('Go to https://mail.google.com and compose a draft')
        self.assertFalse(result.success)
        self.assertEqual(result.final_message, 'Synthetic stop')
        perception.perceive.assert_called_once()
        web.start.assert_not_called()
        self.backend.launch.assert_not_called()
