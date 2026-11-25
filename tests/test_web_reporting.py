import json
import unittest
from unittest.mock import Mock

from core.llm import LLMResponse
from core.web_tasks import run_web_task, requires_download, superseded_release_issue, prerelease_evidence_issue


class WebReportingTests(unittest.TestCase):
    python_goal = ('Use the web agent to visit '
                   '[https://www.python.org/downloads/](https://www.python.org/downloads/). '
                   'Find the latest stable Python 3 release, excluding prereleases. '
                   'Open its release page and report the exact version, release date, '
                   'and whether a Windows installer (64-bit) is available. Include the '
                   'release-page URL. Do not download or install anything. '
                   'If navigation fails, explain the actual error rather than guessing.')

    def test_prerelease_listing_cannot_substitute_for_release_page(self):
        goal = 'Compare newest stable Python release and newest prerelease'
        listing = {'url': 'https://www.python.org/downloads/', 'title': 'Download Python',
                   'text': '3.16 pre-release 2027-10-06 (planned)'}
        stable = {'url': 'https://www.python.org/downloads/release/python-3152/',
                  'title': 'Python Release Python 3.15.2', 'text': 'Windows installer (64-bit)'}
        self.assertTrue(prerelease_evidence_issue(goal, [listing, stable]))
        alpha = {'url': 'https://www.python.org/downloads/release/python-3160a2/',
                 'title': 'Python Release Python 3.16.0a2', 'text': 'Release date'}
        self.assertFalse(prerelease_evidence_issue(goal, [listing, stable, alpha]))
        self.assertTrue(prerelease_evidence_issue(goal, [listing, dict(alpha, title='404 Not Found')]))
        self.assertFalse(prerelease_evidence_issue('Find newest stable Python', [listing, stable]))
        self.assertFalse(prerelease_evidence_issue(goal, [stable]))

    def test_false_comparison_stops_before_verifier_and_learning(self):
        browser = Mock(downloads=[])
        browser.detect_human_gate.return_value = None
        browser.task_observation.return_value = {
            'url': 'https://www.python.org/downloads/', 'title': 'Download Python',
            'text': '3.16 pre-release 2027-10-06 (planned)'}
        provider = Mock()
        provider.call_with_schema.return_value = LLMResponse(
            dict(action='done', target='', text='No exact prerelease exists', reason='Complete',
                 requirements=[dict(description='Compare prerelease', kind='outcome', user_quote='Compare newest prerelease.')]), '')
        ok, message, _ = run_web_task(browser, provider, self.python_goal + ' Compare newest prerelease.',
                                      3, lambda: False, lambda *a: True, lambda _: None)
        self.assertFalse(ok)
        self.assertIn('no exact prerelease release page', message)
        self.assertEqual(provider.call_with_schema.call_count, 2)

    def test_superseded_release_requires_observed_successor(self):
        observed = {'text': 'This release has been superseded by Python 3.16.0a2.'}
        self.assertTrue(superseded_release_issue('Find newest prerelease', observed, []))
        self.assertFalse(superseded_release_issue('Describe this historical release', observed, []))
        self.assertFalse(superseded_release_issue('Find newest prerelease', observed,
                         [{'title': 'Python Release Python 3.16.0a2 | Python.org'}]))
        self.assertTrue(superseded_release_issue('Find newest prerelease', observed,
                        [{'title': 'Python Release Python 3.16.0a20 | Python.org'}]))

    def test_superseded_done_is_rejected_before_model_verification(self):
        browser = Mock(downloads=[])
        browser.detect_human_gate.return_value = None
        browser.task_observation.return_value = {
            'title': 'Python Release Python 3.16.0a1',
            'text': 'This release has been superseded by Python 3.16.0a2.'}
        provider = Mock()
        provider.call_with_schema.return_value = LLMResponse(
            dict(action='done', target='', text='Newest is 3.16.0a1', reason='Complete',
                 requirements=[dict(description='Find newest', kind='outcome', user_quote='Find newest prerelease')]), '')
        ok, message, history = run_web_task(browser, provider, 'Find newest prerelease',
                                           3, lambda: False, lambda *a: True, lambda _: None)
        self.assertFalse(ok)
        self.assertIn('superseded', message)
        self.assertEqual(provider.call_with_schema.call_count, 2)
        self.assertTrue(all(c.kwargs['schema_name'] == 'web_action'
                            for c in provider.call_with_schema.call_args_list))
        browser.task_action.assert_not_called()

    def test_url_path_is_not_a_download_request(self):
        self.assertFalse(requires_download(self.python_goal))
        self.assertFalse(requires_download('Inspect https://example.test/downloads/ and report the version.'))
        self.assertTrue(requires_download('Download the installer from https://example.test/downloads/.'))

    def test_download_prohibitions_do_not_require_receipts(self):
        for goal in ['Compare packages. Do not download anything.',
                     'Report the version without downloading anything.',
                     'Inspect release matrix. No downloads.', 'Never download preview packages.']:
            self.assertFalse(requires_download(goal), goal)
        self.assertTrue(requires_download('Download the stable manifest. Do not download preview packages.'))

    def test_read_only_task_can_finish_without_download(self):
        browser = Mock(downloads=[])
        browser.detect_human_gate.return_value = None
        browser.task_observation.return_value = {'url': 'http://fixture', 'text': 'Compact 7.4'}
        provider = Mock()
        provider.call_with_schema.side_effect = [
            LLMResponse(dict(action='done', target='', text='Compact 7.4', reason='Read',
                requirements=[dict(description='Report version', kind='outcome', user_quote='Report the version')]), ''),
            LLMResponse(dict(complete=True, evidence='Compact 7.4', missing='', requirements=[
                dict(id='R1', status='verified', evidence='Compact 7.4', missing='')]), '')]
        ok, _, _ = run_web_task(browser, provider, 'Report the version',
                                2, lambda: False, lambda *a: True, lambda *a: None)
        self.assertTrue(ok)
        browser.task_action.assert_not_called()

    def run_case(self, complete):
        browser = Mock(downloads=[{'path': 'compact.txt'}], _task_targets={})
        browser.detect_human_gate.return_value = None
        browser.task_observation.side_effect = [
            {'url': 'http://fixture/matrix', 'text': 'Compact 7.4 stable x64 24MB; Full 95MB'},
            {'url': 'http://fixture/compact', 'text': 'Compact download', 'downloads': browser.downloads},
            {'url': 'http://fixture/compact', 'text': 'Compact download', 'downloads': browser.downloads}]
        browser.task_action.return_value = Mock(success=True, message='Navigated')
        answer = 'Compact 7.4 selected; Full exceeds 30MB.'
        action = lambda kind, text: LLMResponse(dict(action=kind, target='', text=text, reason='Continue',
            requirements=[dict(description='Compare and download', kind='outcome', user_quote='Compare and download')]), '')
        verdict = LLMResponse(dict(complete=complete, evidence='Compact download', missing='Unsupported report',
            requirements=[dict(id='R1', status='verified' if complete else 'pending',
                              evidence='Compact download' if complete else '', missing='Unsupported report')]), '')
        provider = Mock()
        provider.call_with_schema.side_effect = [action('navigate', 'http://fixture/compact'),
                                                action('done', answer), verdict,
                                                action('done', answer), verdict]
        result = run_web_task(browser, provider, 'Compare and download; report version and exclusions',
                              5, lambda: False, lambda *a: True, lambda *a: None)
        return result, provider, answer

    def test_report_and_prior_table_reach_verifier(self):
        result, provider, answer = self.run_case(True)
        self.assertTrue(result[0])
        self.assertEqual(result[1], answer)
        actor_prompt = provider.call_with_schema.call_args_list[1].kwargs['user_text']
        verifier_prompt = provider.call_with_schema.call_args_list[2].kwargs['user_text']
        self.assertIn('Full 95MB', actor_prompt)
        self.assertIn('Full 95MB', verifier_prompt)
        self.assertIn(answer, verifier_prompt)
        self.assertIn('claims to verify, not evidence', verifier_prompt)
        self.assertEqual(result[2][-1]['proposed_answer'], answer)
        verifier_call = provider.call_with_schema.call_args_list[2].kwargs
        self.assertIn('Compact download', verifier_call['json_schema']['properties']['evidence']['enum'])
        self.assertIn('not been published yet', verifier_call['system_prompt'])

    def test_report_is_not_automatically_accepted(self):
        result, _, _ = self.run_case(False)
        self.assertFalse(result[0])
        self.assertIn('Unsupported report', result[1])
        self.assertEqual(sum(row['action'] == 'completion_check' for row in result[2]), 2)


if __name__ == '__main__':
    unittest.main()
