import unittest
from unittest.mock import Mock

from core.controller import ExecutionResult
from core.perception import ScreenState, UIElement
from core.planner import StrategicPlanner
from core.reasoning import NextAction
from core.safety import SafetySystem
from core.agent_checks import AttemptGuard, desktop_attempt_signature, observation


class DesktopAttemptIdentityTests(unittest.TestCase):
    def test_alternative_target_or_changed_ui_is_still_allowed(self):
        state = ScreenState('', '', [
            UIElement(1, 'ButtonControl', 'Open', (10, 10, 80, 40), True),
            UIElement(2, 'ButtonControl', 'Open', (10, 50, 80, 80), True),
        ], 'Fixture Editor', 100, 100)
        guard = AttemptGuard()
        first = desktop_attempt_signature(state, NextAction('click', target_tag=1))
        for _ in range(2):
            self.assertFalse(guard.check(observation(state), first))
        self.assertTrue(guard.check(observation(state), first))
        alternative = desktop_attempt_signature(state, NextAction('click', target_tag=2))
        self.assertFalse(guard.check(observation(state), alternative))
        state.elements.append(UIElement(3, 'TextControl', 'New document loaded', (0, 80, 99, 99), True))
        self.assertFalse(guard.check(observation(state), first))

    def test_distinct_text_and_unresolved_tags_remain_distinct(self):
        state = ScreenState('', '', [], 'Fixture Editor', 100, 100)
        self.assertNotEqual(desktop_attempt_signature(state, NextAction('type', text='first')),
                            desktop_attempt_signature(state, NextAction('type', text='second')))
        self.assertNotEqual(desktop_attempt_signature(state, NextAction('click', target_tag=1)),
                            desktop_attempt_signature(state, NextAction('click', target_tag=2)))

    def test_renumbered_same_button_cannot_bypass_loop_limit(self):
        perception, reasoning, controller = Mock(), Mock(), Mock()
        perception.perceive.side_effect = [ScreenState('', '', [
            UIElement(tag, 'ButtonControl', 'Open', (10, 10, 80, 40), True)
        ], 'Fixture Editor', 100, 100) for tag in range(1, 10)]
        reasoning.decide_next_action.side_effect = lambda **kw: NextAction(
            'click', target_tag=kw['screen_state'].elements[0].tag)
        reasoning.replan_after_failure.side_effect = lambda goal, state, history, reason: NextAction(
            'click', target_tag=state.elements[0].tag)
        controller.execute.return_value = ExecutionResult(True, 'Input issued; no observed change')
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
                                   use_planning_phase=False, max_steps=8, step_delay=0,
                                   on_log=lambda _: None)
        result = planner.run('Open the selected document in the fixture editor')
        self.assertFalse(result.success)
        self.assertEqual(controller.execute.call_count, 2)
        self.assertLess(result.steps_taken, 8)


if __name__ == '__main__':
    unittest.main()
