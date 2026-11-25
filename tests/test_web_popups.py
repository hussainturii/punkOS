import unittest
from unittest.mock import Mock
from core.web_agent import WebAgent


class PopupTests(unittest.TestCase):
    def setUp(self):
        self.agent = WebAgent()
        self.first = Mock(url='https://example.test')
        self.first.is_closed.return_value = False
        self.agent._pages = [self.first]

    def popup(self):
        page = Mock()
        page.is_closed.return_value = False
        page.opener.return_value = self.first
        self.agent._register_page(page)
        return page

    def test_single_popup_is_adopted_and_stale_targets_cleared(self):
        self.agent._task_targets = {'1': 'old'}
        page = self.popup()
        self.assertEqual(self.agent._adopt_popup(), '')
        self.assertIs(self.agent.page, page)
        self.assertEqual(self.agent._task_targets, {})
        page.wait_for_load_state.assert_called_once_with('domcontentloaded')

    def test_multiple_popups_are_not_selected_arbitrarily(self):
        self.popup()
        self.popup()
        self.assertIn('ambiguous', self.agent._adopt_popup())
        self.assertIs(self.agent.page, self.first)

    def test_duplicate_registration_has_one_listener(self):
        page = self.popup()
        self.agent._register_page(page)
        page.on.assert_called_once()

    def test_switch_tab_only_uses_current_observed_tab(self):
        self.assertFalse(self.agent.task_action('switch_tab', '99').success)
        page = self.popup()
        self.agent._adopt_popup()
        self.agent._task_tabs = {'0': self.first, '1': page}
        self.assertTrue(self.agent.task_action('switch_tab', '0').success)
        self.assertIs(self.agent.page, self.first)

    def test_tab_limit_is_reported(self):
        self.agent.max_tabs = 1
        page = self.popup()
        self.assertIn('limit', self.agent._adopt_popup())
        page.close.assert_called_once()
