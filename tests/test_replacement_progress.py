import unittest
from unittest.mock import Mock
from core.agent_checks import unfinished_replacement_save
from core.perception import ScreenState, UIElement
from core.reasoning import NextAction


class ReplacementProgressTests(unittest.TestCase):
    goal = 'In Notepad replace every PENDING with READY and save a copy.'

    def state(self, text):
        return ScreenState('', '', [UIElement(1, 'DocumentControl', 'Text editor', (0,0,100,100), True, text=text),
                                    UIElement(2, 'EditControl', 'Find', (0,0,10,10), True, text='PENDING')],
                           'source.txt - Notepad', 100, 100)

    def test_partial_or_unchanged_document_cannot_save(self):
        for text in ('PENDING\nPENDING', 'READY\nPENDING'):
            for key in ('ctrl+s', 'ctrl+shift+s'):
                self.assertTrue(unfinished_replacement_save(self.state(text), NextAction('key', text=key), self.goal))

    def test_completed_document_allowed_despite_find_field(self):
        self.assertFalse(unfinished_replacement_save(self.state('READY\nREADY'), NextAction('key', text='ctrl+shift+s'), self.goal))

    def test_navigation_and_unrelated_goal_unchanged(self):
        self.assertFalse(unfinished_replacement_save(self.state('PENDING'), NextAction('key', text='ctrl+h'), self.goal))
        self.assertFalse(unfinished_replacement_save(self.state('PENDING'), NextAction('key', text='ctrl+s'), 'Save this note'))

    def test_planner_blocks_premature_save(self):
        from core.planner import StrategicPlanner
        from core.safety import SafetySystem
        perception, reasoning, controller = Mock(), Mock(), Mock()
        perception.perceive.return_value = self.state('PENDING')
        reasoning.decide_next_action.return_value = NextAction('key', text='ctrl+shift+s')
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
            use_planning_phase=False, max_steps=1, step_delay=0, on_log=lambda _: None)
        result = planner.run(self.goal)
        self.assertFalse(result.success)
        self.assertIn('Replacement is incomplete', result.history[-1]['outcome'])
        controller.execute.assert_not_called()
