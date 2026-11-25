import unittest
from unittest.mock import Mock
from core.input_evidence import InputEvidence
from core.controller import ExecutionResult
from core.perception import ScreenState, UIElement
from core.reasoning import NextAction
from core.llm import LLMResponse


class InputEvidenceTests(unittest.TestCase):
    def test_missing_runtime_identity_can_use_current_visual_field_evidence(self):
        self.state.elements = []
        self.state.raw_screenshot_b64 = 'current-raw-image'
        self.provider.call_with_schema.return_value = LLMResponse(dict(
            field_found=True, readable=True, visible_text=self.action.text), '')
        receipt = self.guard.observe(self.state, self.provider)
        self.assertIn('NOT verified', receipt)
        request = self.provider.call_with_schema.call_args.kwargs
        self.assertEqual(request['image_b64'], 'current-raw-image')
        self.assertNotIn(self.action.text, request['user_text'])
        self.assertIn('Do not substitute', request['user_text'])
        self.assertFalse(self.guard.pending)

    def test_missing_field_visual_ambiguity_does_not_unlock_typing(self):
        self.state.elements = []
        self.provider.call_with_schema.return_value = LLMResponse(dict(
            field_found=False, readable=False, visible_text=''), '')
        self.guard.observe(self.state, self.provider)
        self.guard.observe(self.state, self.provider)
        self.provider.call_with_schema.assert_called_once()
        self.assertTrue(self.guard.restriction(self.action))

    def test_missing_field_changed_dimensions_do_not_use_stale_region(self):
        self.state.elements = []
        self.state.width = 200
        self.guard.observe(self.state, self.provider)
        self.provider.call_with_schema.assert_not_called()
        self.assertTrue(self.guard.restriction(self.action))

    def test_unannotated_field_crop_is_used_without_expected_answer(self):
        self.state.field_images = {1: 'field-crop'}
        self.provider.call_with_schema.return_value = LLMResponse(dict(
            field_found=True, readable=True, visible_text=''), '')
        self.guard.observe(self.state, self.provider)
        request = self.provider.call_with_schema.call_args.kwargs
        self.assertEqual(request['image_b64'], 'field-crop')
        self.assertNotIn(self.action.text, request['user_text'])
        self.assertIn('empty field', self.guard.detail)
        self.assertTrue(self.guard.restriction(self.action))

    def test_readback_diagnostics_distinguish_missing_unreadable_and_different(self):
        for found, readable, value, expected in [
            (False, False, '', 'could not locate'),
            (True, False, '', 'could not read'),
            (True, True, 'other', 'different text')]:
            with self.subTest(expected=expected):
                self.guard.visual_attempted = False
                self.provider.call_with_schema.return_value = LLMResponse(dict(
                    field_found=found, readable=readable, visible_text=value), '')
                self.guard.observe(self.state, self.provider)
                self.assertIn(expected, self.guard.detail)
                self.assertTrue(self.guard.restriction(self.action))

    def test_planner_blocks_replay_after_successful_click(self):
        from core.planner import StrategicPlanner
        from core.safety import SafetySystem
        perception, reasoning, controller = Mock(), Mock(), Mock()
        perception.perceive.return_value = self.state
        reasoning.provider.supports_vision.return_value = False
        reasoning.decide_next_action.side_effect = [self.action, NextAction('click', target_tag=1), self.action]
        controller.execute.side_effect = [ExecutionResult(False, 'Readback unavailable', True),
                                          ExecutionResult(True, 'Clicked')]
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
            use_planning_phase=False, max_steps=3, step_delay=0, on_log=lambda _: None)
        result = planner.run('Enter the requested value in the current form')
        self.assertFalse(result.success)
        self.assertEqual(controller.execute.call_count, 2)
        self.assertIn('Typing blocked', result.history[-1]['outcome'])

    def setUp(self):
        self.state = ScreenState('image', '', [UIElement(1, 'ComboBoxControl', 'Recipient',
            (1, 1, 100, 20), True, runtime_id=(7,))], 'Form', 100, 100, 42)
        self.guard = InputEvidence()
        self.action = NextAction('type', target_tag=1, text='person@example.test')
        self.guard.record(self.action, self.state, ExecutionResult(False, 'unavailable', True))
        self.provider = Mock()
        self.provider.supports_vision.return_value = True

    def test_click_and_tag_changes_do_not_allow_retry(self):
        self.assertTrue(self.guard.restriction(self.action))
        self.guard.record(NextAction('click', target_tag=1), self.state, ExecutionResult(True, 'Clicked'))
        self.state.elements[0].tag = 9
        self.assertTrue(self.guard.restriction(NextAction('type', target_tag=9, text='other')))
        self.assertFalse(self.guard.restriction(NextAction('wait')))

    def test_visual_readback_is_blind_to_expected_text_and_does_not_verify_acceptance(self):
        self.provider.call_with_schema.return_value = LLMResponse(dict(
            field_found=True, readable=True, visible_text='person@example.test'), '')
        receipt = self.guard.observe(self.state, self.provider)
        self.assertIn('NOT verified', receipt)
        request = self.provider.call_with_schema.call_args.kwargs
        self.assertNotIn(self.action.text, request['user_text'] + request['system_prompt'])
        self.assertFalse(self.guard.restriction(self.action))

    def test_visual_mismatch_is_bounded_and_keeps_typing_blocked(self):
        self.provider.call_with_schema.return_value = LLMResponse(dict(
            field_found=True, readable=True, visible_text='different'), '')
        for _ in range(5):
            self.assertIsNone(self.guard.observe(self.state, self.provider))
        self.provider.call_with_schema.assert_called_once()
        self.assertTrue(self.guard.restriction(self.action))

    def test_sensitive_screen_and_changed_window_never_send_image(self):
        self.state.sensitive_content_detected = True
        self.guard.observe(self.state, self.provider)
        self.state.sensitive_content_detected = False
        self.state.foreground_window_handle = 99
        self.guard.observe(self.state, self.provider)
        self.provider.call_with_schema.assert_not_called()

    def test_accessibility_readback_needs_no_model_call(self):
        self.state.elements[0].text = self.action.text
        self.assertIn('accessibility', self.guard.observe(self.state, self.provider))
        self.provider.call_with_schema.assert_not_called()

    def test_nonvision_and_cancellation_keep_guard(self):
        self.provider.supports_vision.return_value = False
        self.guard.observe(self.state, self.provider)
        self.assertIn('vision is disabled', self.guard.detail)
        self.provider.supports_vision.return_value = True
        self.guard.observe(self.state, self.provider, lambda: True)
        self.provider.call_with_schema.assert_not_called()
        self.assertTrue(self.guard.restriction(self.action))

    def test_provider_error_exposes_diagnostic_without_retrying(self):
        self.provider.call_with_schema.return_value = LLMResponse(None, '', 'Synthetic failure')
        self.guard.observe(self.state, self.provider)
        self.assertIn('provider request failed', self.guard.detail)
        self.guard.observe(self.state, self.provider)
        self.provider.call_with_schema.assert_called_once()
