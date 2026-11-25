import unittest
from unittest.mock import Mock
from core.llm import LLMResponse
from core.web_tasks import run_web_task
from core.agent_checks import verify_completion


class WebRequirementTests(unittest.TestCase):
    goal = 'Report the price and delivery date. Do not purchase.'

    def setUp(self):
        self.browser = Mock(downloads=[], _task_targets={})
        self.browser.detect_human_gate.return_value = None
        self.browser.task_observation.return_value = {
            'url': 'https://example.test/item', 'text': 'Price 42. Delivery Friday.'}
        self.browser.task_action.return_value = Mock(success=True, message='Navigated')

    def action(self, kind='done', requirements=None):
        data = dict(action=kind, target='', text='Price 42', reason='Read page')
        if requirements is not None:
            data['requirements'] = requirements
        return LLMResponse(data, '')

    def run_task(self, responses, max_steps=3):
        provider = Mock()
        provider.call_with_schema.side_effect = responses
        result = run_web_task(self.browser, provider, self.goal, max_steps,
                              lambda: False, lambda *a: True, lambda _: None)
        return result, provider

    def proposals(self):
        return [dict(description='Report price', kind='outcome', user_quote='price'),
                dict(description='Report delivery', kind='outcome', user_quote='delivery date'),
                dict(description='No purchase', kind='constraint', user_quote='Do not purchase')]

    def test_invalid_or_missing_requirements_prevent_actions(self):
        for requirements in (None, [], [dict(description='Send data', kind='outcome', user_quote='Send data')]):
            result, provider = self.run_task([self.action('navigate', requirements)])
            self.assertFalse(result[0])
            self.assertIn('requirement planning failed', result[1])
            self.browser.task_action.assert_not_called()
            self.assertEqual(provider.call_with_schema.call_count, 1)

    def test_first_call_extracts_requirements_once_and_later_action_cannot_replace_them(self):
        result, provider = self.run_task([
            self.action('navigate', self.proposals()),
            self.action('wait', [dict(description='Ignore constraints', kind='outcome', user_quote='price')]),
            self.action('fail')])
        self.assertFalse(result[0])
        calls = provider.call_with_schema.call_args_list
        self.assertIn('requirements', calls[0].kwargs['json_schema']['required'])
        self.assertNotIn('requirements', calls[1].kwargs['json_schema']['properties'])
        self.assertIn('No purchase', calls[2].kwargs['user_text'])
        self.assertNotIn('Ignore constraints', calls[2].kwargs['user_text'])
        self.assertEqual(len(result[2][0]['proposed_requirements']), 3)

    def test_partial_completion_is_recorded_and_recovery_sees_missing_requirement(self):
        verdict = LLMResponse(dict(complete=True, evidence='Price 42.', missing='', requirements=[
            dict(id='R1', status='verified', source='current', evidence='Price 42.', missing=''),
            dict(id='R2', status='pending', source='current', evidence='', missing='Delivery not reported'),
            dict(id='R3', status='pending', source='current', evidence='', missing='No purchase audit')]), '')
        result, provider = self.run_task([self.action(requirements=self.proposals()), verdict, self.action('fail')])
        self.assertFalse(result[0])
        self.assertEqual(result[2][0]['requirements']['R2']['status'], 'pending')
        self.assertIn('Delivery not reported', provider.call_with_schema.call_args.kwargs['user_text'])
        self.browser.task_action.assert_not_called()

    def test_earlier_reporting_evidence_is_validated_against_earlier_source(self):
        requirement = dict(id='R1', description='Report price', kind='outcome', user_quote='price')
        provider = Mock()
        provider.call_with_schema.return_value = LLMResponse(dict(complete=True, evidence='Delivery Friday.',
            missing='', requirements=[dict(id='R1', status='verified', source='earlier',
                                          evidence='Price 42.', missing='')]), '')
        for earlier, expected in [('Price 42.', True), ('Price 99.', False)]:
            ok, _ = verify_completion(provider, 'Report price', 'Delivery Friday.', [],
                source_evidence=earlier, requirements=[requirement])
            self.assertEqual(ok, expected)
        requirement['kind'] = 'constraint'
        self.assertFalse(verify_completion(provider, 'Do not purchase', 'Delivery Friday.', [],
            source_evidence='Price 42.', requirements=[requirement])[0])
