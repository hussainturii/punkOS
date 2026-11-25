import json
import unittest
from unittest.mock import Mock
from core.capabilities import desktop_capabilities, web_capabilities
from core.perception import ScreenState, UIElement
from core.reasoning import ReasoningCore, NEXT_ACTION_SCHEMA
from core.web_tasks import SCHEMA
from core.llm import LLMResponse


class CapabilityTests(unittest.TestCase):
    def test_requested_screenshot_reaches_normal_and_recovery_actions(self):
        provider = Mock()
        provider.supports_vision.return_value = True
        provider.vision_for_actions = False
        provider.call_with_schema.return_value = LLMResponse(dict(
            action_type='wait', target_tag=None, text='', reasoning='Inspect', is_high_risk=False), '')
        core = ReasoningCore(provider)
        goal = 'Inspect the selection using the screenshot if needed.'
        core.decide_next_action(goal, self.state(), [])
        core.replan_after_failure(goal, self.state(), [], 'Selection unclear')
        for call in provider.call_with_schema.call_args_list:
            self.assertEqual(call.kwargs['image_b64'], 'image')
        self.assertIsNone(core._maybe_image(self.state(), 'Do not use screenshots.'))
        state = self.state()
        state.sensitive_content_detected = True
        self.assertIsNone(core._maybe_image(state, goal))
        provider.supports_vision.return_value = False
        self.assertIsNone(core._maybe_image(self.state(), goal))

    def test_fallback_only_vision_keeps_routine_actions_text_only(self):
        from unittest.mock import patch
        from core.llm import build_provider
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'synthetic-test-key'}):
            provider = build_provider({'llm_provider': 'gemini', 'gemini': {
                'use_vision': True, 'vision_for_actions': False}})
        self.assertTrue(provider.supports_vision())
        self.assertIsNone(ReasoningCore(provider)._maybe_image(self.state()))

    def state(self):
        return ScreenState('image', '', [
            UIElement(1, 'ButtonControl', 'Open', (0,0,10,10), True),
            UIElement(2, 'EditControl', 'Input', (0,0,0,0), True, focus_only=True),
            UIElement(3, 'ButtonControl', 'Disabled', (0,0,10,10), False),
            UIElement(4, 'TextControl', 'Label', (0,0,10,10), True, is_interactive=False),
        ], 'Sample app', 100, 100, accessibility_diagnostic='node_limit=True')

    def test_desktop_capabilities_match_observed_targets(self):
        context = desktop_capabilities(self.state(), NEXT_ACTION_SCHEMA, 'image')
        data = json.loads(context.splitlines()[1])
        self.assertEqual(data['clickable_tags'], [1])
        self.assertEqual(data['editable_tags'], [2])
        self.assertFalse(data['coordinate_click_action'])
        self.assertFalse(data['plugin_execution_action'])
        self.assertTrue(data['screenshot_attached'])
        self.assertEqual(data['observation_limits'], 'node_limit=True')

    def test_no_image_does_not_advertise_visual_observation(self):
        context = desktop_capabilities(self.state(), NEXT_ACTION_SCHEMA, None)
        self.assertFalse(json.loads(context.splitlines()[1])['screenshot_attached'])

    def test_web_capabilities_do_not_offer_desktop_or_arbitrary_script(self):
        context = web_capabilities(SCHEMA)
        self.assertIn('"desktop_input": false', context)
        self.assertIn('"arbitrary_javascript": false', context)

    def test_normal_and_recovery_decisions_receive_capabilities(self):
        provider = Mock()
        provider.supports_vision.return_value = True
        provider.call_with_schema.return_value = LLMResponse(dict(
            action_type='wait', target_tag=None, text='', reasoning='Observe', is_high_risk=False), '')
        core = ReasoningCore(provider)
        core.decide_next_action('Inspect current window', self.state(), [])
        core.replan_after_failure('Inspect current window', self.state(), [], 'Control missing')
        self.assertEqual(provider.call_with_schema.call_count, 2)
        for call in provider.call_with_schema.call_args_list:
            self.assertIn('ACTIVE CAPABILITIES', call.kwargs['user_text'])
            self.assertEqual(call.kwargs['image_b64'], 'image')

    def test_sensitive_screen_never_attaches_image(self):
        provider = Mock()
        provider.supports_vision.return_value = True
        state = self.state()
        state.sensitive_content_detected = True
        self.assertIsNone(ReasoningCore(provider)._maybe_image(state))

    def test_planning_receives_actual_capabilities(self):
        provider = Mock()
        provider.supports_vision.return_value = False
        provider.call_with_schema.return_value = LLMResponse(dict(
            summary='Inspect', feasible=True, steps=[], requirements=[]), '')
        ReasoningCore(provider).plan_task('Inspect current window', self.state())
        self.assertIn('ACTIVE CAPABILITIES', provider.call_with_schema.call_args.kwargs['user_text'])
        self.assertIsNone(provider.call_with_schema.call_args.kwargs['image_b64'])
