import unittest
from unittest.mock import Mock, patch
from core.agent_checks import AttemptGuard, verify_completion
from core.llm import LLMResponse, OllamaProvider
from core.reasoning import ReasoningCore
from core.router import route, ExecutionMode
from core.skills import SkillLibrary
from core.web_tasks import run_web_task


class RecoveryTests(unittest.TestCase):
    def test_desktop_verifier_receives_proposed_report(self):
        from core.planner import StrategicPlanner
        from core.reasoning import NextAction
        from core.perception import ScreenState, UIElement
        from core.safety import SafetySystem
        perception, reasoning, controller = Mock(), Mock(), Mock()
        perception.perceive.return_value = ScreenState('', '', [
            UIElement(1, 'DocumentControl', 'Text editor', (0,0,10,10), True, text='Owner: Sara')
        ], 'reviewed.txt - Notepad', 100, 100)
        reasoning.decide_next_action.return_value = NextAction('done', text='The owner is Sara.')
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
            use_planning_phase=False, max_steps=1, step_delay=0, on_log=lambda _: None)
        with patch('core.planner.verify_completion', return_value=(True, 'Owner: Sara')) as verify:
            result = planner.run('Read the current note and report the owner')
        self.assertTrue(result.success)
        self.assertEqual(verify.call_args.kwargs['proposed_answer'], 'The owner is Sara.')
        controller.execute.assert_not_called()

    def test_terminal_failure_records_the_current_observation(self):
        from core.planner import StrategicPlanner
        from core.reasoning import NextAction
        from core.perception import ScreenState, UIElement
        from core.safety import SafetySystem
        perception, reasoning, controller = Mock(), Mock(), Mock()
        perception.perceive.return_value = ScreenState('', '', [
            UIElement(1, 'ButtonControl', 'Close', (0, 0, 20, 20), True)
        ], 'Open', 100, 100)
        reasoning.decide_next_action.return_value = NextAction('fail', text='Missing filename control', reasoning='Only Close observed')
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
                                   use_planning_phase=False, max_steps=3, step_delay=0,
                                   on_log=lambda _: None)
        result = planner.run('Open a synthetic document')
        self.assertFalse(result.success)
        self.assertEqual(result.history[-1]['action'], 'agent_fail')
        self.assertIn('ButtonControl: Close', result.history[-1]['observation'])
        self.assertEqual(result.history[-1]['reasoning'], 'Only Close observed')
        controller.execute.assert_not_called()

    def test_web_rejects_prose_urls_without_navigation(self):
        from core.web_agent import WebAgent
        browser = WebAgent()
        browser.navigate = Mock()
        for url in ('Visit http://127.0.0.1:8766/index.html', 'try running Windows Network Diagnostics',
                    'javascript:alert(1)', 'https://', 'https://example.com:bad'):
            self.assertFalse(browser.task_action('navigate', text=url).success)
        browser.navigate.assert_not_called()
        browser.task_action('navigate', text='http://127.0.0.1:8766/index.html')
        browser.navigate.assert_called_once_with('http://127.0.0.1:8766/index.html')

    def test_web_changed_errors_still_exhaust_recovery(self):
        from core.web_agent import WebResult
        browser = Mock(downloads=[], _task_targets={})
        browser.detect_human_gate.return_value = None
        browser.task_observation.side_effect = [{'title': str(i)} for i in range(10)]
        browser.task_action.return_value = WebResult(False, 'Invalid URL')
        provider = Mock()
        provider.call_with_schema.side_effect = [LLMResponse(
            {'action': 'navigate', 'target': '', 'text': f'wrong prose {i}', 'reason': 'retry',
             'requirements': [dict(description='Visit site', kind='outcome', user_quote='Visit the site')]}, '') for i in range(10)]
        success, message, history = run_web_task(browser, provider, 'Visit the site', 15, lambda: False, lambda *a: True, lambda _: None)
        self.assertFalse(success)
        self.assertIn('three consecutive', message)
        self.assertEqual(browser.task_action.call_count, 3)
        self.assertEqual(len(history), 3)
        self.assertNotIn('\\"observation\\"', provider.call_with_schema.call_args.kwargs['user_text'])

    def test_mouse_requirement_beats_browser_route(self):
        self.assertEqual(route('open chrome and download Python using mouse interaction').mode, ExecutionMode.DESKTOP)
        self.assertEqual(route('search online for Python').mode, ExecutionMode.WEB)

    def test_loop_guard_counts_revisited_states(self):
        guard = AttemptGuard()
        for _ in range(2):
            self.assertFalse(guard.check('A', 'alt+tab'))
            self.assertFalse(guard.check('B', 'alt+tab'))
        self.assertTrue(guard.check('A', 'alt+tab'))
        self.assertFalse(guard.check('A', 'click:search'))

    def test_skills_include_alternatives_and_full_outcome(self):
        skill = SkillLibrary().context('send a WhatsApp message')
        self.assertIn('outgoing message', skill)
        self.assertIn('read-only container', skill)

    def test_missing_evidence_rejects_done(self):
        provider = Mock()
        provider.call_with_schema.return_value = LLMResponse({'complete': True, 'evidence': 'Message sent', 'missing': ''}, '')
        self.assertFalse(verify_completion(provider, 'Send Hey', 'WhatsApp is open', [])[0])

    def test_failed_click_cannot_turn_into_success(self):
        provider = Mock()
        self.assertFalse(verify_completion(provider, 'Send Hey', 'WhatsApp', [{'success': False}])[0])
        provider.call_with_schema.assert_not_called()

    def test_verified_redundancy_block_allows_independent_verification(self):
        provider=Mock()
        provider.call_with_schema.return_value=LLMResponse({'complete':True,'evidence':'verified output','missing':''},'')
        blocked={'action':'key:ctrl+n','success':False,'record_type':'verified_redundancy_block'}
        self.assertTrue(verify_completion(provider,'Create a file','verified output',[{'action':'save','success':True},blocked])[0])
        provider.call_with_schema.assert_called_once()
        provider.reset_mock()
        self.assertFalse(verify_completion(provider,'Create a file','verified output',[{'action':'save','success':False},blocked])[0])
        provider.call_with_schema.assert_not_called()

    def test_verified_quote_is_required(self):
        provider = Mock()
        provider.call_with_schema.return_value = LLMResponse({'complete': True, 'evidence': 'Result is 4', 'missing': ''}, '')
        self.assertTrue(verify_completion(provider, 'Calculate 2+2', 'Calculator: Result is 4', [])[0])

    def test_rejected_quote_can_be_rechecked_without_repeating_input(self):
        provider = Mock()
        provider.call_with_schema.side_effect = [
            LLMResponse({'complete': True, 'evidence': 'The answer is 503', 'missing': ''}, ''),
            LLMResponse({'complete': True, 'evidence': 'Display is 503', 'missing': ''}, ''),
        ]
        observed = 'Calculator\nTextControl: Display is 503 '
        history = [{'action': 'click Equals', 'success': True}]
        ok, reason = verify_completion(provider, 'Calculate 27 * 14 + 125', observed, history)
        self.assertFalse(ok)
        self.assertIn('did not match', reason)
        history.append({'action': 'completion_check', 'success': False, 'outcome': reason})
        self.assertTrue(verify_completion(provider, 'Calculate 27 * 14 + 125', observed, history)[0])
        self.assertEqual(provider.call_with_schema.call_count, 2)
        self.assertNotIn('completion_check', provider.call_with_schema.call_args.kwargs['user_text'])

    def test_rejected_check_does_not_hide_a_real_input_failure(self):
        provider = Mock()
        history = [{'action': 'click Send', 'success': False},
                   {'action': 'completion_check', 'success': False}]
        self.assertFalse(verify_completion(provider, 'Send Hey', 'WhatsApp', history)[0])
        provider.call_with_schema.assert_not_called()

    def test_verifier_does_not_receive_pre_action_screens_or_actor_claims(self):
        provider = Mock()
        provider.call_with_schema.return_value = LLMResponse(
            {'complete': True, 'evidence': 'Display is 503', 'missing': ''}, '')
        history = [{'action': 'click Equals', 'outcome': 'Input sent', 'success': True,
                    'observation': 'Display is 125', 'reasoning': 'Unverified actor claim'}]
        self.assertTrue(verify_completion(provider, 'Calculate 27 * 14 + 125',
                        'Calculator\nDisplay is 503', history)[0])
        prompt = provider.call_with_schema.call_args.kwargs['user_text']
        self.assertNotIn('Display is 125', prompt)
        self.assertNotIn('Unverified actor claim', prompt)
        self.assertIn('click Equals', prompt)
        self.assertTrue(prompt.endswith('Calculator\nDisplay is 503'))

    def test_visible_quote_does_not_override_incomplete_goal(self):
        provider = Mock()
        provider.call_with_schema.return_value = LLMResponse(
            {'complete': False, 'evidence': 'Draft Hey', 'missing': 'Message not sent'}, '')
        self.assertFalse(verify_completion(provider, 'Send Hey', 'Draft Hey', [])[0])

    def test_planner_rechecks_completion_without_repeating_desktop_input(self):
        from core.planner import StrategicPlanner
        from core.reasoning import NextAction
        from core.perception import ScreenState, UIElement
        from core.safety import SafetySystem
        screen = ScreenState('', '', [UIElement(1, 'TextControl', 'Display is 503',
                            (0, 0, 20, 20), True)], 'Calculator', 100, 100)
        for second_quote, expected_success in [('Display is 503', True), ('Answer 503', False)]:
            with self.subTest(second_quote=second_quote):
                perception, reasoning, controller = Mock(), Mock(), Mock()
                perception.perceive.return_value = screen
                reasoning.decide_next_action.return_value = NextAction('done', text='Result is 503')
                reasoning.replan_after_failure.return_value = NextAction('done', text='Result is 503')
                reasoning.provider.call_with_schema.side_effect = [
                    LLMResponse({'complete': True, 'evidence': quote, 'missing': ''}, '')
                    for quote in ('Answer 503', second_quote)
                ]
                planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
                    use_planning_phase=False, max_steps=15, step_delay=0, on_log=lambda _: None)
                result = planner.run('In Calculator calculate 27 * 14 + 125')
                self.assertEqual(result.success, expected_success)
                self.assertEqual(result.steps_taken, 2)
                self.assertEqual(reasoning.provider.call_with_schema.call_count, 2)
                controller.execute.assert_not_called()

    def test_literal_document_text_is_not_changed(self):
        for text in ('Please open the window and enter the text for the meeting.', 'ctrl+s', '  indented\n'):
            action = ReasoningCore._parse_next_action({'action_type': 'type', 'text': text})
            self.assertEqual(action.action_type, 'type')
            self.assertEqual(action.text, text)

    def test_model_tag_must_match(self):
        self.assertFalse(OllamaProvider._model_present('qwen2.5:7b', {'qwen2.5', 'qwen2.5:3b'}))

    def test_opaque_app_stops_without_model_call(self):
        from core.perception import ScreenState, UIElement
        provider = Mock()
        provider.supports_vision.return_value = False
        state = ScreenState('', '', [UIElement(1, 'DocumentControl', 'WhatsApp', (0,0,100,100), True, is_interactive=False)], 'WhatsApp', 100, 100)
        action = ReasoningCore(provider).decide_next_action('Send Hey', state, [])
        self.assertEqual(action.action_type, 'fail')
        self.assertEqual(action.raw['blocked_by'], 'perception')
        provider.call_with_schema.assert_not_called()

    def test_web_navigation_not_download_success(self):
        browser = Mock(downloads=[])
        browser.detect_human_gate.return_value = None
        browser.task_observation.return_value = {'title': 'Python'}
        provider = Mock()
        provider.call_with_schema.return_value = LLMResponse({'action': 'done', 'target': '', 'text': 'Downloaded', 'reason': 'done',
            'requirements': [dict(description='Download', kind='outcome', user_quote='download Python')]}, '')
        success, message, history = run_web_task(browser, provider, 'download Python', 5, lambda: False, lambda *a: True, lambda x: None)
        self.assertFalse(success)
        self.assertIn('receipt', message)
        self.assertEqual(provider.call_with_schema.call_count, 2)

    def test_web_stop_after_inference_prevents_action(self):
        browser = Mock(downloads=[])
        browser.detect_human_gate.return_value = None
        browser.task_observation.return_value = {'title': 'Example'}
        provider = Mock()
        provider.call_with_schema.return_value = LLMResponse({'action': 'click', 'target': '1', 'text': '', 'reason': 'click'}, '')
        stop = Mock(side_effect=[False, True])
        self.assertFalse(run_web_task(browser, provider, 'click', 5, stop, lambda *a: True, lambda x: None)[0])
        browser.task_action.assert_not_called()

    def test_planner_stops_successful_input_loop(self):
        from core.planner import StrategicPlanner
        from core.reasoning import NextAction
        from core.controller import ExecutionResult
        from core.perception import ScreenState, UIElement
        from core.safety import SafetySystem
        screen = ScreenState('', '', [UIElement(1, 'ButtonControl', 'OK', (0,0,20,20), True)], 'Fixture', 100, 100)
        perception = Mock()
        perception.perceive.return_value = screen
        reasoning = Mock()
        action = NextAction('click', target_tag=1)
        reasoning.decide_next_action.return_value = action
        reasoning.replan_after_failure.return_value = action
        controller = Mock()
        controller.execute.return_value = ExecutionResult(True, 'Input issued')
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
                                   use_planning_phase=False, max_steps=80, step_delay=0,
                                   on_log=lambda _:None)
        result = planner.run('Click the target and finish the task')
        self.assertFalse(result.success)
        self.assertEqual(controller.execute.call_count, 2)
        self.assertLess(result.steps_taken, 10)
        self.assertIsNone(reasoning.decide_next_action.call_args.kwargs['current_step_idx'])


if __name__ == '__main__':
    unittest.main()
