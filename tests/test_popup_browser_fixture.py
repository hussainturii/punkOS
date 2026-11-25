"""Local headless browser fixture; no external website or model calls."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from core.web_agent import WebAgent


class FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = (b'<html><title>Listing</title><body><a href="/details" target="_blank">Details</a></body></html>'
                if self.path == '/' else b'<html><title>Details</title><body>Fixture information</body></html>')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class PopupBrowserFixtureTests(unittest.TestCase):
    def test_target_blank_is_observed_and_can_return_to_original_tab(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logs = []
        agent = WebAgent(headless=True, on_log=logs.append)
        try:
            result = agent.start()
            self.assertTrue(result.success, result.message)
            self.assertTrue(agent.navigate(f'http://127.0.0.1:{server.server_port}/').success)
            initial = agent.task_observation()
            target = next(str(row['index']) for row in initial['targets'] if row['text'] == 'Details')
            self.assertTrue(agent.task_action('click', target).success)
            observed = agent.task_observation()
            self.assertEqual(observed['title'], 'Details')
            self.assertEqual(observed['navigation'].get('status'), 200, (logs, agent._navigation_receipts, observed))
            self.assertEqual(len(observed['tabs']), 2)
            self.assertTrue(agent.task_action('switch_tab', '0').success)
            self.assertEqual(agent.task_observation()['title'], 'Listing')
        finally:
            agent.close()
            server.shutdown()
            server.server_close()
            thread.join(3)
