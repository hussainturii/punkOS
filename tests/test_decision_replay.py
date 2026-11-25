import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks import decision_replay
from core.llm import LLMResponse


class CloudReplayTests(unittest.TestCase):
    def test_cloud_provider_needs_no_ollama_readiness_method(self):
        class Provider:
            def name(self): return 'gemini/fixture'
            def supports_vision(self): return False
            def call_with_schema(self, **kwargs):
                if 'Calculator' in kwargs['user_text']:
                    value = dict(action_type='click', target_tag=1, text='', reasoning='Multiply', is_high_risk=False)
                else:
                    value = dict(action_type='type', target_tag=1, text='Hello benchmark', reasoning='Type', is_high_risk=False)
                return LLMResponse(value, '')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            with patch.object(decision_replay, 'build_provider', return_value=Provider()) as build:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = decision_replay.main(['--provider', 'gemini', '--output', str(path)])
            self.assertEqual(result, 0)
            self.assertEqual(build.call_args.args[0]['llm_provider'], 'gemini')
            self.assertTrue(all(r['passed'] for r in json.loads(path.read_text())['results']))
