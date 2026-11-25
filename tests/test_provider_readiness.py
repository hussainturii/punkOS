import unittest
from unittest.mock import Mock, patch
from core.llm import OllamaProvider
from core.planner import StrategicPlanner


class ReadinessTests(unittest.TestCase):
    def provider(self):
        p = object.__new__(OllamaProvider)
        p.host = 'http://localhost:11434'
        p.text_model = 'qwen2.5:3b'
        p.use_vision = False
        p.cancel_check = lambda: False
        return p

    @patch('core.inference_transport.get_json')
    def test_checks_exact_model_with_short_timeout(self, get):
        get.return_value = {'models': [{'name': 'qwen2.5:3b'}]}
        p = self.provider()
        p.ensure_available()
        self.assertEqual(get.call_args.args[1], 3.0)
        get.return_value = {'models': [{'name': 'qwen2.5:7b'}]}
        with self.assertRaises(RuntimeError):
            p.ensure_available()

    def test_offline_service_prevents_handoff_and_input(self):
        p = self.provider()
        p.ensure_available = Mock(side_effect=ConnectionError('offline'))
        reasoning = Mock(provider=p)
        perception, controller, handoff = Mock(), Mock(), Mock()
        planner = StrategicPlanner(perception, reasoning, controller, Mock(), desktop_handoff=handoff, on_log=lambda _: None)
        result = planner.run('Open Notepad and type Hello')
        self.assertFalse(result.success)
        self.assertEqual(result.steps_taken, 0)
        handoff.assert_not_called()
        perception.perceive.assert_not_called()
        controller.execute.assert_not_called()
