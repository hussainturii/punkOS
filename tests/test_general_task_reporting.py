"""General multi-outcome acceptance fixtures, without UI or provider calls."""
import unittest
from unittest.mock import Mock
from core.task_report import partial_report
from core.reasoning import TaskPlan, NextAction
from core.controller import ExecutionResult
from core.perception import ScreenState, UIElement
from core.planner import StrategicPlanner
from core.safety import SafetySystem


class GeneralReportingTests(unittest.TestCase):
    def requirements(self, first='Create output', second='Preserve original'):
        return [dict(id='R1', description=first, kind='outcome'),
                dict(id='R2', description=second, kind='constraint')]

    def test_varied_partial_work_never_becomes_complete(self):
        cases = [
            ('Editor', 'Create a separate document', 'Keep the source unchanged', 'Source comparison unavailable'),
            ('Browser', 'Read first product price', 'Compare the second product', 'Second page failed'),
            ('Form', 'Enter destination', 'Confirm accepted selection', 'Selection unreadable'),
            ('Files', 'Locate source', 'Verify output contents', 'Output missing'),
            ('Spreadsheet', 'Read input cells', 'Verify formula results', 'Formula results unavailable'),
        ]
        for name, first, second, reason in cases:
            with self.subTest(module=name):
                requirements = self.requirements(first, second)
                history = [dict(action='completion_check', success=False,
                    proposed_requirements=requirements, requirements={
                        'R1': dict(status='verified', evidence='Observed source value'),
                        'R2': dict(status='pending', missing=reason)})]
                report = partial_report('Task stopped', False, None, history)
                self.assertTrue(report.startswith('Task stopped'))
                self.assertIn('task remains incomplete', report)
                self.assertIn('Previously verified', report)
                self.assertIn(first, report)
                self.assertIn(reason, report)

    def test_successful_input_receipts_do_not_count_as_verified_outcomes(self):
        plan = TaskPlan('Work', [], True, self.requirements())
        report = partial_report('Timeout', False, plan, [dict(action='type', success=True, outcome='Input sent')])
        self.assertIn('No requested outcome has a recorded successful verification', report)
        self.assertIn('R1:', report)
        self.assertIn('R2:', report)

    def test_latest_assessment_replaces_earlier_verified_state(self):
        plan = TaskPlan('Work', [], True, self.requirements())
        history = [dict(action='completion_check', requirements={'R1': dict(status='verified', evidence='Old result')}),
                   dict(action='completion_check', requirements={'R1': dict(status='pending', missing='Changed afterward')})]
        report = partial_report('Stopped', False, plan, history)
        self.assertNotIn('Previously verified', report)
        self.assertIn('Changed afterward', report)

    def test_verified_status_without_evidence_is_not_reported_as_verified(self):
        plan = TaskPlan('Work', [], True, self.requirements())
        report = partial_report('Stopped', False, plan,
            [dict(action='completion_check', requirements={'R1': dict(status='verified')})])
        self.assertNotIn('Previously verified', report)

    def test_success_and_unplanned_errors_preserve_existing_messages(self):
        self.assertEqual(partial_report('Completed', True, None, []), 'Completed')
        self.assertEqual(partial_report('Provider unavailable', False, None, []), 'Provider unavailable')

    def test_early_failure_reports_unverified_work_consistently_in_ui_and_history(self):
        perception, reasoning, controller, store = Mock(), Mock(), Mock(), Mock()
        perception.perceive.return_value = ScreenState('', '', [
            UIElement(1, 'ButtonControl', 'Details', (0,0,10,10), True)], 'Workspace', 100,100)
        reasoning.plan_task.return_value = TaskPlan('Inspect and compare', [], True, self.requirements())
        reasoning.decide_next_action.side_effect = [NextAction('wait'), NextAction('fail', text='Required control unavailable')]
        controller.execute.return_value = ExecutionResult(True, 'Waited')
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
            history_store=store, use_planning_phase=True, max_steps=3, step_delay=0, on_log=lambda _: None)
        result = planner.run('Inspect the current workspace and compare values')
        self.assertFalse(result.success)
        self.assertIn('Required control unavailable', result.final_message)
        self.assertIn('No requested outcome has a recorded successful verification', result.final_message)
        self.assertEqual(store.add_task.call_args.args[0].final_message, result.final_message)
        self.assertEqual(controller.execute.call_count, 1)
