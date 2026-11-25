import unittest
from unittest.mock import Mock

from core.web_agent import WebAgent


class WebTabTests(unittest.TestCase):
    def setUp(self):
        self.agent = WebAgent()
        self.pages = [Mock(), Mock(), Mock()]
        self.agent._pages = list(self.pages)
        self.agent._active_index = 1

    def test_closing_earlier_tab_preserves_active_page(self):
        self.assertTrue(self.agent.close_tab(0).success)
        self.assertIs(self.agent.page, self.pages[1])
        self.assertEqual(self.agent._active_index, 0)

    def test_failed_close_keeps_tab_and_selection(self):
        self.pages[0].close.side_effect = RuntimeError('Browser disconnected')
        self.assertFalse(self.agent.close_tab(0).success)
        self.assertEqual(self.agent._pages, self.pages)
        self.assertIs(self.agent.page, self.pages[1])

    def test_failed_switch_keeps_previous_selection(self):
        self.pages[2].bring_to_front.side_effect = RuntimeError('Page closed')
        self.assertFalse(self.agent.switch_tab(2).success)
        self.assertIs(self.agent.page, self.pages[1])

    def test_closing_active_last_tab_selects_remaining_neighbor(self):
        self.agent._active_index = 2
        self.assertTrue(self.agent.close_tab(2).success)
        self.assertIs(self.agent.page, self.pages[1])

    def test_last_tab_cannot_be_closed(self):
        self.agent._pages = [self.pages[0]]
        self.agent._active_index = 0
        self.assertFalse(self.agent.close_tab(0).success)
        self.pages[0].close.assert_not_called()


if __name__ == '__main__':
    unittest.main()
