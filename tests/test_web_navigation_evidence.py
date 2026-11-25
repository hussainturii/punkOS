import unittest
from unittest.mock import Mock
from core.web_agent import WebAgent


class NavigationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.agent = WebAgent()
        self.page = Mock(url='https://example.test/destination')
        self.agent._pages = [self.page]
        self.agent._track_navigation(self.page)
        self.callback = self.page.on.call_args.args[1]

    def response(self, status=200, main=True, navigation=True):
        result = Mock(status=status, url=self.page.url)
        result.request.is_navigation_request.return_value = navigation
        result.request.frame = self.page.main_frame if main else Mock()
        return result

    def test_only_main_document_final_response_is_recorded(self):
        for response in (self.response(302), self.response(404, main=False), self.response(500, navigation=False)):
            self.callback(response)
            self.assertEqual(self.agent._current_navigation(), {})
        self.callback(self.response(404))
        self.assertEqual(self.agent._current_navigation()['status'], 404)
        self.callback(self.response(200))
        self.assertEqual(self.agent._current_navigation()['status'], 200)
        self.assertEqual(self.agent._current_navigation()['sequence'], 2)

    def test_different_visible_document_cannot_use_old_receipt(self):
        self.callback(self.response())
        self.page.url = 'https://example.test/other'
        self.assertEqual(self.agent._current_navigation(), {})

    def test_fragment_does_not_discard_same_document_receipt(self):
        self.callback(self.response())
        self.page.url += '#section'
        self.assertEqual(self.agent._current_navigation()['status'], 200)
