import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

from core.inference_transport import post_json, post_response, InferenceCancelled
from core.task_policy import needs_plan


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.arrived = threading.Event()
        self.release = threading.Event()
        arrived, release = self.arrived, self.release
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                arrived.set()
                release.wait(3)
                try:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"message":{"content":"{}"}}')
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/api/chat'

    def tearDown(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_cancel_before_dispatch_sends_nothing(self):
        with self.assertRaises(InferenceCancelled):
            post_json(self.url, {}, 3, lambda: True)
        self.assertFalse(self.arrived.is_set())

    def test_cancel_while_waiting_for_first_byte(self):
        stop = threading.Event()
        def cancel():
            self.arrived.wait(2)
            stop.set()
        worker = threading.Thread(target=cancel)
        worker.start()
        start = time.monotonic()
        with self.assertRaises(InferenceCancelled):
            post_json(self.url, {}, 5, stop.is_set)
        worker.join()
        self.assertLess(time.monotonic() - start, 2)
        self.assertTrue(self.arrived.is_set())

    def test_wall_clock_timeout_without_response(self):
        with self.assertRaises((TimeoutError, __import__('httpx').TimeoutException)):
            post_json(self.url, {}, 0.2, lambda: False)

    def test_success_returns_response(self):
        self.release.set()
        self.assertEqual(post_json(self.url, {}, 3, lambda: False), {'message': {'content': '{}'}})

    def test_cloud_transport_cancels_wait_and_allows_next_request(self):
        import httpx
        stop = threading.Event()
        worker = threading.Thread(target=lambda: (self.arrived.wait(2), stop.set()))
        worker.start()
        start = time.monotonic()
        with self.assertRaises(InferenceCancelled):
            post_response(self.url, {}, {'x-test-key': 'fake'}, httpx.Timeout(5), stop.is_set)
        worker.join()
        self.assertTrue(self.arrived.is_set())
        self.assertLess(time.monotonic() - start, 2)
        self.release.set()
        response = post_response(self.url, {}, {}, httpx.Timeout(3), lambda: False)
        self.assertEqual(response.status_code, 200)

    def test_cloud_cancel_before_dispatch_sends_nothing(self):
        import httpx
        with self.assertRaises(InferenceCancelled):
            post_response(self.url, {}, {}, httpx.Timeout(3), lambda: True)
        self.assertFalse(self.arrived.is_set())

    def test_http_success_inside_playwright_sync_context(self):
        import asyncio
        import httpx
        from playwright.sync_api import sync_playwright
        self.release.set()
        with sync_playwright():
            self.assertTrue(asyncio.get_running_loop().is_running())
            response = post_response(self.url, {}, {}, httpx.Timeout(3), lambda: False)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(post_json(self.url, {}, 3, lambda: False), {'message': {'content': '{}'}})

    def test_cloud_cancel_and_recover_inside_playwright_sync_context(self):
        import httpx
        from playwright.sync_api import sync_playwright
        stop = threading.Event()
        worker = threading.Thread(target=lambda: (self.arrived.wait(2), stop.set()))
        with sync_playwright():
            worker.start()
            start = time.monotonic()
            with self.assertRaises(InferenceCancelled):
                post_response(self.url, {}, {}, httpx.Timeout(5), stop.is_set)
            worker.join()
            self.assertTrue(self.arrived.is_set())
            self.assertLess(time.monotonic() - start, 2)
            self.release.set()
            self.assertEqual(post_response(self.url, {}, {}, httpx.Timeout(3), lambda: False).status_code, 200)


class PlannerLifecycleTests(unittest.TestCase):
    def make_planner(self, **kwargs):
        from core.planner import StrategicPlanner
        from core.perception import ScreenState, UIElement
        from core.reasoning import NextAction
        self.reasoning = Mock()
        self.reasoning.decide_next_action.return_value = NextAction('fail', text='fixture end')
        self.perception = Mock()
        self.perception.perceive.return_value = ScreenState('', '', [UIElement(1, 'ButtonControl', 'OK', (0,0,20,20), True)], 'Calculator', 100, 100)
        self.controller = Mock()
        return StrategicPlanner(self.perception, self.reasoning, self.controller, Mock(), on_log=lambda _: None, **kwargs)

    def test_stop_before_worker_dispatch_is_preserved(self):
        planner = self.make_planner()
        planner.request_stop()
        self.assertFalse(planner.run('Use Calculator').success)
        self.perception.perceive.assert_not_called()
        self.reasoning.plan_task.assert_not_called()

    def test_late_done_after_stop_cannot_be_success(self):
        from core.reasoning import NextAction
        planner = self.make_planner(use_planning_phase=False)
        logs = []
        planner.on_log = logs.append
        def late_done(*a, **k):
            planner.request_stop()
            return NextAction('done', text='Finished')
        self.reasoning.decide_next_action.side_effect = late_done
        result = planner.run('Use Calculator')
        self.assertFalse(result.success)
        self.assertEqual(result.final_message, 'User stopped task')
        self.assertTrue(any('User stopped task' in line for line in logs))
        self.assertFalse(any('Hit max steps' in line for line in logs))
        self.controller.execute.assert_not_called()
        self.reasoning.provider.call_with_schema.assert_not_called()

    def test_auto_calculator_skips_upfront_plan(self):
        planner = self.make_planner(planning_mode='auto')
        planner.run('Use Calculator to calculate 27 times 14 then add 125')
        self.reasoning.plan_task.assert_not_called()
        self.reasoning.decide_next_action.assert_called_once()

    def test_planning_policy_keeps_document_and_complex_tasks(self):
        self.assertTrue(needs_plan('Open Notepad and type a summary and save it'))
        self.assertTrue(needs_plan('Use Calculator to calculate totals and save a file'))
        self.assertTrue(needs_plan('Compare these reports'))

    def test_deadline_rejects_late_action(self):
        from core.reasoning import NextAction
        planner = self.make_planner(use_planning_phase=False)
        def expire(*a, **k):
            planner._deadline = time.monotonic() - 1
            return NextAction('click', target_tag=1)
        self.reasoning.decide_next_action.side_effect = expire
        result = planner.run('Use Calculator')
        self.assertFalse(result.success)
        self.assertEqual(result.final_message, 'Task time limit exceeded')
        self.controller.execute.assert_not_called()


class ConfirmationTests(unittest.TestCase):
    def test_result_presentation_distinguishes_stops_and_failures(self):
        from core.task_policy import result_presentation
        for message in ('User stopped task', 'User cancelled high-risk action.',
                        'Task time limit exceeded'):
            self.assertEqual(result_presentation(False, message)[0], 'stopped')
        self.assertEqual(result_presentation(False, 'Execution error')[0], 'failed')
        self.assertEqual(result_presentation(True, 'Verified')[0], 'completed')

    def test_reply_delivered_without_worker_event_loop(self):
        from ui.dashboard import AgentWorker
        from core.reasoning import NextAction
        from core.safety import RiskAssessment, RiskLevel
        from PyQt6.QtCore import Qt
        planner = Mock()
        planner.is_stopped.return_value = False
        worker = AgentWorker(planner, 'fixture')
        seen = threading.Event()
        worker.confirm_signal.connect(lambda _: seen.set(), Qt.ConnectionType.DirectConnection)
        results = []
        thread = threading.Thread(target=lambda: results.append(worker._wait_for_confirmation(
            NextAction('click'), RiskAssessment(RiskLevel.HIGH, 'test', requires_confirmation=True))))
        thread.start()
        self.assertTrue(seen.wait(2))
        worker.confirm_response_received.emit(True)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [True])

    def test_stop_prevents_confirmation_request(self):
        from ui.dashboard import AgentWorker
        planner = Mock()
        planner.is_stopped.return_value = True
        self.assertFalse(AgentWorker(planner, 'fixture')._wait_for_confirmation(None, None))


if __name__ == '__main__':
    unittest.main()
