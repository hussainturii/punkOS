import unittest
from unittest.mock import Mock, patch
from core.llm import LLMResponse
from core.web_tasks import run_web_task


class WebRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.browser = Mock(downloads=[], _task_targets={})
        self.browser.detect_human_gate.return_value = None
        self.browser.task_observation.return_value = {'url': 'https://example.test', 'title': 'Page', 'text': 'Observed text'}
        self.browser.task_action.return_value = Mock(success=True, message='Navigated')
        self.provider = Mock()

    def action(self, kind='navigate'):
        return LLMResponse(dict(action=kind, target='', text='https://example.test', reason='Inspect',
            requirements=[dict(description='Report', kind='outcome', user_quote='Report')]), '')

    def run_task(self):
        return run_web_task(self.browser, self.provider, 'Report the page', 4,
                            lambda: False, lambda *args: True, lambda _: None)

    def test_invalid_action_does_not_reach_browser(self):
        for payload in (['navigate'], {'action': 'navigate'}, {'action': 'shell', 'target': '', 'text': '', 'reason': ''}):
            self.provider.call_with_schema.return_value = LLMResponse(payload, '')
            self.assertFalse(self.run_task()[0])
        self.browser.task_action.assert_not_called()

    def test_observation_error_stops_before_inference(self):
        self.browser.task_observation.side_effect = RuntimeError('Page closed')
        result = self.run_task()
        self.assertFalse(result[0])
        self.assertIn('Page closed', result[1])
        self.provider.call_with_schema.assert_not_called()

    def test_action_exception_is_not_replayed(self):
        self.provider.call_with_schema.return_value = self.action()
        self.browser.task_action.side_effect = RuntimeError('Connection lost')
        result = self.run_task()
        self.assertFalse(result[0])
        self.assertIn('effects are uncertain', result[1])
        self.browser.task_action.assert_called_once()

    def test_transport_retry_does_not_repeat_action(self):
        error = LLMResponse(None, '', 'gemini request failed (RemoteProtocolError).')
        self.provider.call_with_schema.side_effect = [error, self.action(), self.action('fail')]
        self.run_task()
        self.browser.task_action.assert_called_once()
        calls = self.provider.call_with_schema.call_args_list
        self.assertEqual(calls[0].kwargs, calls[1].kwargs)

    def test_transport_retry_budget_is_shared(self):
        error = LLMResponse(None, '', 'gemini request failed (RemoteProtocolError).')
        self.provider.call_with_schema.side_effect = [error, self.action(), error]
        self.assertFalse(self.run_task()[0])
        self.assertEqual(self.provider.call_with_schema.call_count, 3)
        self.browser.task_action.assert_called_once()

    def test_failed_navigation_is_not_source_evidence_or_completion(self):
        self.browser.task_action.return_value = Mock(success=False, message='HTTP 404')
        self.browser.task_observation.side_effect = [
            {'url': 'https://example.test', 'text': 'Listing'},
            {'url': 'https://example.test/missing', 'text': 'ERROR_PAGE_SENTINEL'},
            {'url': 'https://example.test/missing', 'text': 'ERROR_PAGE_SENTINEL'}]
        self.provider.call_with_schema.side_effect = [self.action(), self.action('done'), self.action('done')]
        result = self.run_task()
        self.assertFalse(result[0])
        self.assertIn('HTTP 404', result[1])
        request = self.provider.call_with_schema.call_args_list[1].kwargs['user_text']
        earlier = request.split('EARLIER SOURCE OBSERVATIONS')[1].split('OBSERVATION:')[0]
        self.assertNotIn('ERROR_PAGE_SENTINEL', earlier)

    def test_successful_click_navigation_clears_error_before_completion(self):
        self.browser.task_action.side_effect = [Mock(success=False, message='HTTP 404'),
                                               Mock(success=True, message='Clicked')]
        self.browser.task_observation.side_effect = [
            {'url': 'https://example.test', 'text': 'Listing'},
            {'url': 'https://example.test/missing', 'text': 'Error',
             'navigation': {'sequence': 1, 'status': 404, 'url': 'https://example.test/missing'}},
            {'url': 'https://example.test/recovered', 'text': 'Recovered',
             'navigation': {'sequence': 2, 'status': 200, 'url': 'https://example.test/recovered'}}]
        self.provider.call_with_schema.side_effect = [self.action(), self.action('click'), self.action('done')]
        with patch('core.web_tasks.verify_completion', return_value=(True, 'Verified')) as verify:
            result = self.run_task()
        self.assertTrue(result[0])
        self.assertIn('Recovered', verify.call_args.args[2])

    def test_http_error_from_click_blocks_completion(self):
        self.browser.task_observation.return_value['navigation'] = {
            'sequence': 1, 'status': 500, 'url': 'https://example.test'}
        self.provider.call_with_schema.side_effect = [self.action('done'), self.action('done')]
        with patch('core.web_tasks.verify_completion') as verify:
            result = self.run_task()
        self.assertFalse(result[0])
        self.assertIn('HTTP 500', result[1])
        verify.assert_not_called()
