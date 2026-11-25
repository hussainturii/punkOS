"""Real Qt panel and headless browser; no Gemini or external pages."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import threading
import time
import unittest
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from core.web_agent import WebAgent
from ui.web_agent_view import WebAgentPanel


class WebPanelThreadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_idle(self, panel):
        deadline = time.monotonic() + 20
        while panel._busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertFalse(panel._busy, 'Panel operation timed out')

    def test_shutdown_waits_for_running_operation_without_terminating_thread(self):
        from unittest.mock import Mock
        from core.web_agent import WebResult
        arrived, release = threading.Event(), threading.Event()
        agent = Mock(is_running=True, form_fields=[])
        agent.generated_script.return_value = ''
        agent.screenshot_b64.return_value = ''
        agent.list_tabs.return_value = []
        agent.detect_human_gate.return_value = None
        def navigate(*args):
            arrived.set()
            release.wait(5)
            return WebResult(True, 'Navigated')
        agent.navigate.side_effect = navigate
        agent.close.return_value = WebResult(True, 'Closed')
        panel = WebAgentPanel(agent)
        try:
            panel._dispatch('navigate', 'about:blank')
            self.assertTrue(arrived.wait(2))
            panel._dispatch('new_tab')
            agent.new_tab.assert_not_called()
            self.assertFalse(panel.shutdown(10))
            agent.close.assert_not_called()
            release.set()
            self.assertTrue(panel.shutdown(3000))
            agent.close.assert_called_once()
        finally:
            release.set()
            panel.shutdown(5000)
            self.app.processEvents()
            panel.deleteLater()

    def test_repeated_operations_refresh_and_shutdown_share_owner(self):
        agent = WebAgent(headless=True, navigation_timeout_ms=3000)
        calls = []
        for name in ('start', 'navigate', 'new_tab', 'switch_tab', 'dom_snapshot',
                     'discover_form_fields', 'screenshot_b64', 'list_tabs',
                     'detect_human_gate', 'generated_script', 'close'):
            original = getattr(agent, name)
            def tracked(*args, _name=name, _original=original, **kwargs):
                calls.append((_name, threading.get_ident()))
                return _original(*args, **kwargs)
            setattr(agent, name, tracked)
        panel = WebAgentPanel(agent)
        results = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                page = Path(tmp) / 'fixture.html'
                page.write_text('<title>Thread fixture</title><h1>Panel test</h1><input aria-label="Query">')
                panel._dispatch('navigate', page.as_uri(), then=results.append)
                self.wait_idle(panel)
                self.assertTrue(results[-1].success, results[-1].message)
                self.assertEqual(panel.tab_list.count(), 1)
                self.assertTrue(panel._snapshot['preview'])
                panel._on_discover_fields()
                self.wait_idle(panel)
                self.assertEqual(panel.field_table.rowCount(), 1)
                panel._on_refresh_dom()
                self.wait_idle(panel)
                self.assertEqual(panel.dom_tree.topLevelItemCount(), 1)
                from core.web_agent import WebResult
                panel._render_scrape(WebResult(True, 'Fixture', [{'title': 'Panel test'}]))
                self.assertTrue(panel.export_json_btn.isEnabled())
                panel._dispatch('new_tab', then=results.append)
                self.assertEqual(panel.dom_tree.topLevelItemCount(), 0)
                self.assertEqual(panel.field_table.rowCount(), 0)
                self.assertEqual(panel.scrape_table.rowCount(), 0)
                self.assertFalse(panel.export_json_btn.isEnabled())
                self.assertFalse(panel.export_csv_btn.isEnabled())
                self.wait_idle(panel)
                self.assertTrue(results[-1].success, results[-1].message)
                self.assertEqual(panel._snapshot['fields'], [])
                self.assertEqual(panel.tab_list.count(), 2)
                panel._dispatch('switch_tab', 0, then=results.append)
                self.wait_idle(panel)
                self.assertTrue(results[-1].success, results[-1].message)
                panel._dispatch('switch_tab', 99, then=results.append)
                self.wait_idle(panel)
                self.assertFalse(results[-1].success)
                panel._on_close_browser()
                self.wait_idle(panel)
                self.assertEqual(panel.tab_list.count(), 0)
                panel._dispatch('navigate', page.as_uri(), then=results.append)
                self.wait_idle(panel)
                self.assertTrue(results[-1].success, results[-1].message)
        finally:
            self.assertTrue(panel.shutdown(15000))
            self.app.processEvents()
            panel.deleteLater()
        owners = {thread for _, thread in calls}
        self.assertEqual(len(owners), 1, calls)
        self.assertNotIn(threading.get_ident(), owners)
        self.assertEqual(calls[-1][0], 'close')


if __name__ == '__main__':
    unittest.main()
