import unittest
from unittest.mock import Mock
from core.agent_checks import requested_editor_replacement
from core.perception import ScreenState, UIElement
from core.reasoning import NextAction
from core.planner import StrategicPlanner
from core.safety import SafetySystem
from core.controller import ExecutionResult


class ReplacementRiskTests(unittest.TestCase):
    goal = 'In Notepad replace every PENDING with READY.'

    def state(self):
        return ScreenState('', '', [
            UIElement(1, 'ButtonControl', 'Replace', (0,0,10,10), True),
            UIElement(2, 'DocumentControl', 'Text editor', (0,0,10,10), True, text='PENDING'),
            UIElement(3, 'EditControl', 'Find', (0,0,10,10), True, text='PENDING'),
            UIElement(4, 'EditControl', 'Replace', (0,0,10,10), True, text='READY')],
            'source.txt - Notepad', 100,100)

    def test_only_matching_requested_editor_fields_qualify(self):
        state = self.state(); action = NextAction('click', target_tag=1)
        self.assertTrue(requested_editor_replacement(state, action, self.goal))
        state.elements[-1].text = 'WRONG'
        self.assertFalse(requested_editor_replacement(state, action, self.goal))
        state = self.state(); state.foreground_window_title = 'Confirm Save As'
        self.assertFalse(requested_editor_replacement(state, action, self.goal))
        state = self.state(); state.elements[-1].is_enabled = False
        self.assertFalse(requested_editor_replacement(state, action, self.goal))

    def test_planner_skips_false_confirmation_but_preserves_model_flag(self):
        for flagged in (False, True):
            with self.subTest(flagged=flagged):
                perception, reasoning, controller, confirm = Mock(), Mock(), Mock(), Mock(return_value=False)
                perception.perceive.return_value = self.state()
                reasoning.decide_next_action.return_value = NextAction('click', target_tag=1, is_high_risk=flagged)
                controller.execute.return_value = ExecutionResult(True, 'clicked')
                planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
                    use_planning_phase=False, max_steps=1, step_delay=0, on_log=lambda _: None, on_confirm=confirm)
                planner.run(self.goal)
                self.assertEqual(confirm.call_count, int(flagged))
                self.assertEqual(controller.execute.call_count, int(not flagged))


if __name__ == '__main__':
    unittest.main()
