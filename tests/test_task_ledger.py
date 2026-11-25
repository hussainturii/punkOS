import unittest
from unittest.mock import Mock

from core.task_ledger import TaskLedger
from core.llm import LLMResponse
from core.web_agent import WebResult
from core.web_tasks import run_web_task


class TaskLedgerTests(unittest.TestCase):
    def test_prior_observations_are_bounded_distinct_from_current_and_not_receipts(self):
        ledger = TaskLedger()
        for i in range(7):
            ledger.observe(f'Actual field value {i}', [dict(action='type', outcome='Invented claim')]*i)
        prior = ledger.prior_observations()
        self.assertEqual(len(ledger.observations), 4)
        self.assertIn('Actual field value 5', prior)
        self.assertNotIn('Actual field value 6', prior)
        self.assertNotIn('Actual field value 0', prior)
        self.assertNotIn('Invented claim', prior)
        ledger.observe('Actual field value 6', [])
        self.assertEqual(len(ledger.observations), 4)

    def test_rejected_completion_is_not_an_executed_action_and_survives_other_failures(self):
        ledger = TaskLedger()
        ledger.observe('screen', [])
        history = [{'action': 'completion_check', 'success': False,
                    'outcome': 'Second requested document has not been verified',
                    'observation': 'screen'}]
        ledger.observe('screen', history)
        self.assertIsNone(ledger.latest)
        for i in range(12):
            history.append({'action': f'click {i}', 'success': False,
                            'outcome': 'Missing target', 'observation': 'screen'})
            ledger.observe('screen', history)
        self.assertIn('Second requested document', ledger.context())

    def test_desktop_recovery_receives_shared_evidence(self):
        from core.planner import StrategicPlanner
        from core.reasoning import NextAction
        from core.perception import ScreenState, UIElement
        from core.safety import SafetySystem
        from core.controller import ExecutionResult
        perception, reasoning, controller = Mock(), Mock(), Mock()
        perception.perceive.return_value = ScreenState('', '', [
            UIElement(1, 'ButtonControl', 'Details', (0, 0, 20, 20), True)
        ], 'Workspace', 100, 100)
        reasoning.decide_next_action.side_effect = [
            NextAction('key', text='tab'), NextAction('key', text='shift+tab')]
        reasoning.replan_after_failure.return_value = NextAction('fail', text='Control unavailable')
        controller.execute.return_value = ExecutionResult(False, 'Target does not accept focus')
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
                                   use_planning_phase=False, max_steps=3, step_delay=0,
                                   on_log=lambda _: None)
        result = planner.run('Inspect the current window and report its details')
        self.assertFalse(result.success)
        self.assertEqual(controller.execute.call_count, 2)
        context = reasoning.replan_after_failure.call_args.args[3]
        self.assertIn('TASK EXECUTION EVIDENCE', context)
        self.assertIn('Target does not accept focus', context)
        self.assertIn('"observation_changed": false', context)

    def test_failed_approach_survives_recent_history_window(self):
        ledger = TaskLedger()
        ledger.observe('initial', [])
        history = [{'action': 'click panel', 'success': False,
                    'outcome': 'read-only container', 'observation': 'initial'}]
        ledger.observe('initial', history)
        for i in range(12):
            history.append({'action': 'navigate', 'success': True,
                            'outcome': 'input sent', 'observation': str(i)})
            ledger.observe(str(i+1), history)
        self.assertIn('read-only container', ledger.context())
        self.assertEqual(list(ledger.failures.values()), [1])

    def test_receipt_success_is_separate_from_observed_change(self):
        ledger = TaskLedger()
        ledger.observe('screen', [])
        history = [{'action': 'type', 'success': True, 'observation': 'screen'}]
        ledger.observe('screen', history)
        self.assertTrue(ledger.latest['receipt_success'])
        self.assertFalse(ledger.latest['observation_changed'])
        self.assertEqual(ledger.unchanged, 1)
        # No new action must not increment the counter.
        ledger.observe('screen', history)
        self.assertEqual(ledger.unchanged, 1)
        history.append({'action': 'focus', 'success': True, 'observation': 'screen'})
        ledger.observe('new screen', history)
        self.assertEqual(ledger.unchanged, 0)
        self.assertNotIn('complete', ledger.latest)

    def test_failures_are_bounded_and_counted_without_recounting_history(self):
        ledger = TaskLedger()
        history = []
        for i in range(20):
            history.append({'action': str(i), 'success': False, 'outcome': 'x'*1000})
            ledger.observe('screen', history)
        self.assertEqual(len(ledger.failures), 8)
        self.assertLess(len(ledger.context()), 6500)
        ledger.observe('screen', history)
        self.assertTrue(all(n == 1 for n in ledger.failures.values()))

    def test_browser_receives_failure_cause_without_extra_inference(self):
        browser = Mock(downloads=[], _task_targets={})
        browser.detect_human_gate.return_value = None
        browser.task_observation.return_value = {'title': 'Search', 'text': 'Search field'}
        browser.task_action.return_value = WebResult(False, 'Target disappeared')
        provider = Mock()
        provider.call_with_schema.side_effect = [
            LLMResponse({'action': 'click', 'target': '2', 'text': '', 'reason': 'Open',
                         'requirements': [dict(description='Inspect', kind='outcome', user_quote='Inspect the page')]}, ''),
            LLMResponse({'action': 'fail', 'target': '', 'text': 'Cannot locate target', 'reason': 'Stop'}, '')]
        ok, _, _ = run_web_task(browser, provider, 'Inspect the page', 4,
                               lambda: False, lambda *a: True, lambda _: None)
        self.assertFalse(ok)
        self.assertEqual(provider.call_with_schema.call_count, 2)
        prompt = provider.call_with_schema.call_args.kwargs['user_text']
        self.assertIn('TASK EXECUTION EVIDENCE', prompt)
        self.assertIn('Target disappeared', prompt)
        self.assertIn('"observation_changed": false', prompt)


if __name__ == '__main__':
    unittest.main()
