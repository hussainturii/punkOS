import unittest
from unittest.mock import Mock
from core.agent_checks import observed_progress, redundant_calculator_input
from core.perception import ScreenState, UIElement
from core.reasoning import NextAction
from core.reasoning import ReasoningCore
from core.llm import LLMResponse


class ObservedProgressTests(unittest.TestCase):
    def setUp(self):
        self.state = ScreenState('', '', [UIElement(1, 'TextControl', 'Display is 27', (0,0,10,10), True)], 'Calculator', 100, 100)
        self.history = [{'action': "type tag=None text='27'", 'success': True,
                         'observation': 'Calculator\nTextControl: Display is 0 '}]

    def test_current_effect_is_explicit(self):
        message = observed_progress(self.state, self.history)
        self.assertIn('Observed before: TextControl: Display is 0', message)
        self.assertIn('Observed now: TextControl: Display is 27', message)

    def test_schema_excludes_read_only_and_disabled_targets(self):
        self.state.elements[0].is_interactive = False
        self.state.elements += [UIElement(2, 'ButtonControl', 'Multiply by', (0,0,10,10), True),
                                UIElement(3, 'ButtonControl', 'Disabled', (0,0,10,10), False)]
        self.assertEqual(ReasoningCore._action_schema(self.state)['properties']['target_tag']['enum'], [None, 2])

    def test_number_entry_requires_intervening_operation(self):
        self.assertNotIn('type', ReasoningCore._action_schema(self.state, self.history)['properties']['action_type']['enum'])
        self.history.append({'action': "click tag=2 text=''", 'success': True})
        self.assertIn('type', ReasoningCore._action_schema(self.state, self.history)['properties']['action_type']['enum'])

    def test_decision_and_recovery_receive_observed_change(self):
        self.state.elements.append(UIElement(2, 'ButtonControl', 'Multiply by', (0,0,10,10), True))
        provider = Mock()
        provider.supports_vision.return_value = False
        provider.call_with_schema.return_value = LLMResponse(
            {'action_type': 'click', 'target_tag': 2, 'text': '', 'reasoning': 'Multiply', 'is_high_risk': False}, '')
        reasoning = ReasoningCore(provider)
        reasoning.decide_next_action('Calculate in Calculator', self.state, self.history)
        reasoning.replan_after_failure('Calculate in Calculator', self.state, self.history, 'duplicate input')
        for call in provider.call_with_schema.call_args_list:
            self.assertIn('Observed now: TextControl: Display is 27', call.kwargs['user_text'])

    def test_unchanged_screen_does_not_claim_progress(self):
        self.history[0]['observation'] = 'Calculator\nTextControl: Display is 27 '
        self.assertIn('No observable UI change', observed_progress(self.state, self.history))

    def test_duplicate_number_is_blocked_only_after_visible_success(self):
        action = NextAction('type', text='27')
        self.assertTrue(redundant_calculator_input(self.state, action, self.history))
        self.history[-1]['success'] = False
        self.assertFalse(redundant_calculator_input(self.state, action, self.history))

    def test_same_operand_after_operator_is_allowed(self):
        self.history.append({'action': "click tag=20 text=''", 'success': True})
        self.assertFalse(redundant_calculator_input(self.state, NextAction('type', text='27'), self.history))

    def test_rejected_retry_does_not_erase_duplicate_guard(self):
        self.history.append({'action': 'type:None:27', 'success': False, 'outcome': 'Duplicate rejected'})
        self.assertTrue(redundant_calculator_input(self.state, NextAction('type', text='27'), self.history))

    def test_other_input_and_apps_are_not_blocked(self):
        self.assertFalse(redundant_calculator_input(self.state, NextAction('type', text='14'), self.history))
        self.state.foreground_window_title = 'Notepad'
        self.assertFalse(redundant_calculator_input(self.state, NextAction('type', text='27'), self.history))
