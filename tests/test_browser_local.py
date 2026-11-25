"""Real headless browser against a local fixture; no external accounts or model calls."""
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.web_agent import WebAgent


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/missing':
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Error 404: File not Found')
            return
        self.send_response(200)
        if self.path == '/payload':
            self.send_header('Content-Disposition', 'attachment; filename="fixture.txt"')
            body = b'PunkOS local download fixture'
        else:
            self.send_header('Content-Type', 'text/html')
            body = b'<html><body><h1>Local benchmark</h1><input aria-label="Query"><button onclick="document.querySelector(\"h1\").textContent=\"Updated\"">Update</button><a href="/payload">Download fixture</a></body></html>'
        if self.path != '/payload':
            body += b'<a href="/payload" download>Native download</a>'
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class BrowserLocalTests(unittest.TestCase):
    def test_external_close_and_restart(self):
        browser = WebAgent(headless=True, navigation_timeout_ms=5000)
        try:
            result = browser.start()
            self.assertTrue(result.success, result.message)
            browser.page.close()
            result = browser.navigate('about:blank')
            self.assertFalse(result.success)
            self.assertIn('tab is closed', result.message)
            self.assertFalse(browser.start().success)
            self.assertTrue(browser.new_tab().success)
            self.assertEqual(len(browser._pages), 1)
            self.assertFalse(browser.page.is_closed())
            browser._browser.close()
            self.assertFalse(browser.is_running)
            result = browser.start()
            self.assertTrue(result.success, result.message)
            self.assertTrue(browser.is_running)
            self.assertTrue(browser.navigate('about:blank').success)
        finally:
            browser.close()

    def test_observe_fill_download_persist(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory() as tmp:
            browser = WebAgent(headless=True, download_dir=tmp, navigation_timeout_ms=5000)
            try:
                result = browser.start()
                self.assertTrue(result.success, result.message)
                failed = browser.navigate(f'http://127.0.0.1:{server.server_port}/missing')
                self.assertFalse(failed.success)
                self.assertEqual(failed.data['status'], 404)
                self.assertIn('observed link', failed.message)
                self.assertTrue(browser.navigate(f'http://127.0.0.1:{server.server_port}').success)
                observed = browser.task_observation()
                query = next(str(r['index']) for r in observed['targets'] if r['tag'] == 'INPUT')
                self.assertTrue(browser.task_action('fill', query, 'test query').success)
                self.assertEqual(browser.page.locator('input').input_value(), 'test query')
                observed = browser.task_observation()
                target = next(str(r['index']) for r in observed['targets'] if r['text'] == 'Download fixture')
                result = browser.task_action('download', target)
                self.assertTrue(result.success, result.message)
                path = Path(result.data['path'])
                observed = browser.task_observation()
                target = next(str(r['index']) for r in observed['targets'] if r['text'] == 'Native download')
                result2 = browser.task_action('click', target)
                self.assertTrue(result2.success, result2.message)
                self.assertEqual(len(browser.downloads), 2)
                path2 = Path(result2.data['path'])
                browser.close()
                self.assertEqual(path.read_bytes(), b'PunkOS local download fixture')
                self.assertEqual(path2.read_bytes(), b'PunkOS local download fixture')
            finally:
                browser.close()
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    unittest.main()
