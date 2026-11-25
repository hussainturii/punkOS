"""Offline provider contracts. Uses fake keys and mocked HTTP; never calls an API."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from core.cloud_llm import CloudJSONProvider, closed_schema
from core.llm import build_provider
from core.reasoning import NEXT_ACTION_SCHEMA, PLAN_SCHEMA

ACTION = {
    "action_type": "done", "target_tag": None, "text": "Complete",
    "reasoning": "Verified output", "is_high_risk": False,
}


def envelope(provider, value=ACTION, finish=None):
    raw = json.dumps(value)
    if provider == "gemini":
        return {"candidates": [{
            "finishReason": finish or "STOP",
            "content": {"parts": [{"text": raw}]},
        }]}
    return {"choices": [{
        "finish_reason": finish or "stop", "message": {"content": raw},
    }]}


class CloudProviderTests(unittest.TestCase):
    def test_literal_table_whitespace_allowed_only_for_verification(self):
        from core.agent_checks import VERIFY_SCHEMA
        data = envelope('gemini')
        raw = '{"complete":true,"evidence":"Compact\t7.4\nStable","missing":""}'
        data['candidates'][0]['content']['parts'][0]['text'] = raw
        with patch('httpx.post', return_value=httpx.Response(200, json=data)):
            result = self.provider('gemini').call_with_schema('', '', VERIFY_SCHEMA, 'verify_completion', '')
        self.assertIsNone(result.error)
        self.assertEqual(result.parsed_json['evidence'], 'Compact\t7.4\nStable')
        with patch('httpx.post', return_value=httpx.Response(200, json=data)):
            result = self.provider('gemini').call_with_schema('', '', VERIFY_SCHEMA, 'next_action', '')
        self.assertIsNone(result.parsed_json)
        self.assertIn('malformed JSON', result.error)

    def test_verification_does_not_accept_truncated_json(self):
        from core.agent_checks import VERIFY_SCHEMA
        data = envelope('gemini')
        data['candidates'][0]['content']['parts'][0]['text'] = '{"complete":true'
        with patch('httpx.post', return_value=httpx.Response(200, json=data)):
            result = self.provider('gemini').call_with_schema('', '', VERIFY_SCHEMA, 'verify_completion', '')
        self.assertIsNone(result.parsed_json)
        self.assertIn('malformed JSON', result.error)

    def test_cloud_cancellation_returns_no_action(self):
        from core.inference_transport import InferenceCancelled
        provider = self.provider('gemini')
        provider.cancel_check = lambda: True
        with patch('core.cloud_llm.post_response', side_effect=InferenceCancelled):
            result = provider.call_with_schema('', '', NEXT_ACTION_SCHEMA, 'next_action', '')
        self.assertIsNone(result.parsed_json)
        self.assertIn('cancelled', result.error)

    def test_long_explanation_does_not_reject_valid_completion(self):
        completion = dict(ACTION, reasoning=(
            'The saved document was reopened from the requested path and all three '
            'lines are visible with their exact requested content.'))
        self.assertGreater(len(completion['reasoning']), 90)
        result, _ = self.call('gemini', envelope('gemini', completion))
        self.assertIsNone(result.error)
        self.assertEqual(result.parsed_json, completion)

    def test_long_explanation_does_not_relax_action_validation(self):
        invalid = dict(ACTION, action_type='run_arbitrary_code', reasoning='x' * 120)
        result, _ = self.call('gemini', envelope('gemini', invalid))
        self.assertIsNone(result.parsed_json)
        self.assertIn('rule: enum', result.error)

    def test_schema_failure_reports_rule_without_echoing_model_values(self):
        invalid = dict(ACTION, action_type='private-screen-content')
        result, _ = self.call('gemini', envelope('gemini', invalid))
        self.assertIsNone(result.parsed_json)
        self.assertIn('rule: enum', result.error)
        self.assertIn('No action', result.error)
        self.assertNotIn('private-screen-content', result.error)

    def provider(self, name, **kwargs):
        return CloudJSONProvider(name, "fake-secret-never-real", "test-model", **kwargs)

    def call(self, name, data=None, status=200, **kwargs):
        response = httpx.Response(status, json=envelope(name) if data is None else data)
        with patch("httpx.post", return_value=response) as post:
            result = self.provider(name, **kwargs).call_with_schema(
                "System", "Goal", NEXT_ACTION_SCHEMA, "next_action", "Choose action",
                image_b64="fake-image",
            )
        post.assert_called_once()
        return result, post.call_args

    def test_factory_reads_correct_env_and_does_not_call_network(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-g", "XAI_API_KEY": "fake-x"}, clear=True):
            with patch("httpx.post") as post:
                for name in ("gemini", "grok"):
                    provider = build_provider({"llm_provider": name, name: {"model": "custom-model"}})
                    self.assertEqual(provider.name(), f"{name}/custom-model")
                    self.assertFalse(provider.supports_vision())
                post.assert_not_called()

    def test_missing_environment_key_is_actionable(self):
        with patch.dict(os.environ, {}, clear=True):
            for name, env in (("gemini", "GEMINI_API_KEY"), ("grok", "XAI_API_KEY")):
                with self.assertRaisesRegex(RuntimeError, env):
                    build_provider({"llm_provider": name, name: {"api_key": "ignored-file-key"}})

    def test_gemini_request_and_nullable_action(self):
        result, call = self.call("gemini")
        self.assertEqual(result.parsed_json, ACTION)
        self.assertIsNone(result.error)
        self.assertTrue(call.args[0].endswith("/models/test-model:generateContent"))
        self.assertEqual(call.kwargs["headers"]["x-goog-api-key"], "fake-secret-never-real")
        payload = call.kwargs["json"]
        self.assertEqual(payload["contents"][0]["parts"], [{"text": "Goal"}])
        generation = payload["generationConfig"]
        self.assertEqual(generation["responseFormat"]["text"]["mimeType"], "APPLICATION_JSON")
        self.assertEqual(
            generation["responseFormat"]["text"]["schema"]["properties"]["target_tag"]["type"],
            ["integer", "null"],
        )
        self.assertNotIn("temperature", generation)
        self.assertFalse(call.kwargs["follow_redirects"])

    def test_grok_request_and_nullable_action(self):
        result, call = self.call("grok", temperature=0.1, max_tokens=1234)
        self.assertEqual(result.parsed_json, ACTION)
        self.assertEqual(call.args[0], "https://api.x.ai/v1/chat/completions")
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer fake-secret-never-real")
        payload = call.kwargs["json"]
        self.assertEqual(payload["messages"][1]["content"], "Goal")
        self.assertEqual(payload["max_tokens"], 1234)
        self.assertEqual(payload["temperature"], 0.1)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])

    def test_groq_factory_uses_its_own_key_and_namespaced_model(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-groq", "XAI_API_KEY": "fake-xai"}, clear=True):
            provider = build_provider({"llm_provider": "groq"})
            self.assertEqual(provider.name(), "groq/qwen/qwen3.8-27b")
            self.assertEqual(provider._api_key, "fake-groq")
            self.assertEqual(provider.max_tokens, 4096)
        with patch.dict(os.environ, {"XAI_API_KEY": "fake-xai"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
                build_provider({"llm_provider": "groq"})

    def test_groq_contract_and_hidden_reasoning(self):
        provider = CloudJSONProvider("groq", "fake-groq", "qwen/qwen3.8-27b", max_tokens=4096)
        with patch("httpx.post", return_value=httpx.Response(200, json=envelope("groq"))) as post:
            result = provider.call_with_schema("System", "Goal", NEXT_ACTION_SCHEMA, "action", "")
        self.assertEqual(result.parsed_json, ACTION)
        call = post.call_args
        self.assertEqual(call.args[0], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer fake-groq")
        payload = call.kwargs["json"]
        self.assertEqual(payload["model"], "qwen/qwen3.8-27b")
        self.assertEqual(payload["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["reasoning_format"], "hidden")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])

    def test_groq_invalid_output_and_quota_are_not_success(self):
        result, _ = self.call("groq", envelope("groq", {**ACTION, "action_type": "shell"}))
        self.assertIsNone(result.parsed_json)
        result, _ = self.call("groq", {"error": {"message": "Rate limit reached for tokens per minute."}}, 429)
        self.assertIn("groq HTTP 429", result.error)
        self.assertIn("tokens per minute", result.error)

    def test_groq_namespace_validation_rejects_urls_and_path_traversal(self):
        for model in ("qwen/../model", "https://evil.invalid", "/model", "qwen/model?x=1"):
            with self.assertRaises(ValueError):
                CloudJSONProvider("groq", "fake-key", model)

    def test_vision_is_opt_in_for_both(self):
        _, gemini = self.call("gemini", use_vision=True)
        self.assertEqual(gemini.kwargs["json"]["contents"][0]["parts"][1]["inlineData"]["data"], "fake-image")
        _, grok = self.call("grok", use_vision=True)
        self.assertEqual(
            grok.kwargs["json"]["messages"][1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,fake-image",
        )

    def test_schema_is_closed_recursively_without_mutation(self):
        original = copy.deepcopy(PLAN_SCHEMA)
        closed = closed_schema(PLAN_SCHEMA)
        self.assertFalse(closed["additionalProperties"])
        self.assertFalse(closed["properties"]["steps"]["items"]["additionalProperties"])
        self.assertEqual(PLAN_SCHEMA, original)

    def test_invalid_actions_never_reach_executor(self):
        for name in ("gemini", "grok"):
            for value in (
                {**ACTION, "action_type": "execute_shell"},
                {**ACTION, "target_tag": "7"},
                {**ACTION, "is_high_risk": "false"},
                {"action_type": "done"},
                {**ACTION, "extra": True},
                [],
            ):
                with self.subTest(provider=name, value=value):
                    result, _ = self.call(name, envelope(name, value))
                    self.assertIsNone(result.parsed_json)
                    self.assertIsNotNone(result.error)

    def test_truncation_is_rejected_even_with_valid_json(self):
        for name, reason in (("gemini", "MAX_TOKENS"), ("grok", "length")):
            result, _ = self.call(name, envelope(name, finish=reason))
            self.assertIsNone(result.parsed_json)
            self.assertIsNotNone(result.error)

    def test_refusal_and_empty_candidates(self):
        result, _ = self.call("gemini", {"promptFeedback": {"blockReason": "SAFETY"}})
        self.assertIsNotNone(result.error)
        data = envelope("grok")
        data["choices"][0]["message"]["refusal"] = "Cannot comply"
        result, _ = self.call("grok", data)
        self.assertIsNone(result.parsed_json)

    def test_gemini_thought_parts_are_not_parsed_as_output(self):
        data = envelope("gemini")
        data["candidates"][0]["content"]["parts"].insert(0, {"text": "internal note", "thought": True})
        result, _ = self.call("gemini", data)
        self.assertEqual(result.parsed_json, ACTION)

    def test_malformed_provider_envelopes(self):
        for name in ("gemini", "grok"):
            for data in ([], {"candidates": None, "choices": None}, {"choices": []}):
                result, _ = self.call(name, data)
                self.assertIsNone(result.parsed_json)
                self.assertIsNotNone(result.error)

    def test_invalid_json_text_and_non_json_http_body(self):
        for name in ("gemini", "grok"):
            data = envelope(name)
            if name == "gemini":
                data["candidates"][0]["content"]["parts"] = [{"text": "```bad JSON"}]
            else:
                data["choices"][0]["message"]["content"] = "```bad JSON"
            result, _ = self.call(name, data)
            self.assertIsNone(result.parsed_json)
            with patch("httpx.post", return_value=httpx.Response(200, text="<html>bad gateway</html>")):
                result = self.provider(name).call_with_schema("", "", NEXT_ACTION_SCHEMA, "action", "")
            self.assertIsNotNone(result.error)

    def test_http_errors_do_not_echo_secrets_or_retry(self):
        for name in ("gemini", "grok"):
            for code in (400, 401, 403, 404, 429, 500, 302):
                result, _ = self.call(name, {"error": "fake-secret-never-real"}, code)
                self.assertIn(str(code), result.error)
                self.assertNotIn("fake-secret", result.error)
                self.assertEqual(result.raw_text, "")

    def test_xai_error_string_preserves_actionable_reason(self):
        result, _ = self.call("grok", {
            "code": "Some credits are required",
            "error": "Your team has no credits. Add credits in the console.",
        }, 400)
        self.assertIn("Your team has no credits", result.error)
        self.assertIn("Some credits are required", result.error)
        self.assertIn("model test-model", result.error)

    def test_nested_error_preserves_rejected_parameter(self):
        for name in ("grok", "gemini"):
            result, _ = self.call(name, {"error": {
                "message": "Unsupported parameter",
                "param": "max_tokens",
                "type": "invalid_request_error",
                "request": {"Authorization": "do-not-print-this-header"},
            }}, 400)
            self.assertIn("Unsupported parameter", result.error)
            self.assertIn("param: max_tokens", result.error)
            self.assertNotIn("do-not-print", result.error)

    def test_error_redaction_happens_before_truncation(self):
        result, _ = self.call("grok", {"error": {
            "message": (
                "x" * 580 + " fake-secret-never-real "
                "Bearer another-credential xai-abcdefghijk"
            ),
        }}, 400)
        self.assertNotIn("fake-secret", result.error)
        self.assertNotIn("another-credential", result.error)
        self.assertNotIn("xai-abcdefghijk", result.error)
        self.assertIn("[REDACTED]", result.error)
        self.assertLess(len(result.error), 800)

    def test_error_details_remove_terminal_control_characters(self):
        result, _ = self.call("grok", {"error": "Invalid\nrequest\x1b[31m"}, 400)
        self.assertNotIn("\n", result.error)
        self.assertNotIn("\x1b", result.error)

    def test_unstructured_http_errors_do_not_dump_body(self):
        with patch("httpx.post", return_value=httpx.Response(502, text="<html>private proxy data</html>")):
            result = self.provider("grok").call_with_schema("", "", NEXT_ACTION_SCHEMA, "action", "")
        self.assertIn("HTTP 502", result.error)
        self.assertNotIn("private proxy", result.error)
        result, _ = self.call("grok", ["unstructured error"], 400)
        self.assertNotIn("unstructured error", result.error)

    def test_timeout_does_not_echo_secret(self):
        with patch("httpx.post", side_effect=httpx.ReadTimeout("fake-secret-never-real")) as post:
            result = self.provider("grok").call_with_schema("", "", NEXT_ACTION_SCHEMA, "action", "")
        post.assert_called_once()
        self.assertIn("ReadTimeout", result.error)
        self.assertNotIn("fake-secret", result.error)

    def test_connect_timeout_reports_connection_phase_without_retry(self):
        with patch("httpx.post", side_effect=httpx.ConnectTimeout("fake-secret-never-real")) as post:
            result = self.provider("gemini", connect_timeout=45).call_with_schema(
                "", "", NEXT_ACTION_SCHEMA, "action", "",
            )
        post.assert_called_once()
        self.assertIn("exceeded 45s", result.error)
        self.assertIn("--network-check", result.error)
        self.assertIn("validity is unknown", result.error)
        self.assertNotIn("fake-secret", result.error)

    def test_factory_passes_configured_connection_timeout(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}, clear=True):
            default = build_provider({"llm_provider": "gemini"})
            custom = build_provider({"llm_provider": "gemini", "gemini": {"connect_timeout": 45}})
        self.assertEqual(default._timeout.connect, 30)
        self.assertEqual(custom._timeout.connect, 45)

    def test_network_check_needs_no_key_and_sends_no_credentials(self):
        import check_provider
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys, "argv", ["check_provider.py", "--provider", "gemini", "--network-check"]):
                with patch("httpx.get", return_value=httpx.Response(404)) as get:
                    with patch("httpx.post") as post, patch.object(Path, "write_text") as write:
                        with redirect_stdout(io.StringIO()) as output:
                            self.assertEqual(check_provider.main(), 0)
        self.assertEqual(get.call_args.args[0], "https://generativelanguage.googleapis.com/")
        self.assertNotIn("headers", get.call_args.kwargs)
        self.assertNotIn("params", get.call_args.kwargs)
        post.assert_not_called()
        write.assert_not_called()
        self.assertIn("HTTP 404", output.getvalue())
        self.assertIn("remain unverified", output.getvalue())

    def test_network_timeout_hides_exception_and_proxy_credentials(self):
        import check_provider
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://user:secret@proxy.invalid"}, clear=True):
            with patch("httpx.get", side_effect=httpx.ConnectTimeout("private-url-and-key")):
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(check_provider.check_network("gemini", 30), 1)
        self.assertNotIn("secret", output.getvalue())
        self.assertNotIn("private-url", output.getvalue())
        self.assertIn("HTTPS_PROXY", output.getvalue())

    def test_connection_timeout_cli_validation(self):
        import argparse
        import check_provider
        for invalid in ("0", "-1", "nan", "inf"):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_provider.positive_seconds(invalid)
        self.assertEqual(check_provider.positive_seconds("45"), 45)

    def test_connect_timeout_cli_override_reaches_saved_config(self):
        import check_provider
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}, clear=True):
            with patch.object(sys, "argv", [
                "check_provider.py", "--provider", "gemini", "--connect-timeout", "45", "--save",
            ]):
                with patch.object(Path, "read_text", return_value='{"llm_provider":"ollama"}'):
                    with patch.object(Path, "write_text") as write, redirect_stdout(io.StringIO()):
                        self.assertEqual(check_provider.main(), 0)
        self.assertEqual(json.loads(write.call_args.args[0])["gemini"]["connect_timeout"], 45)

    def test_model_id_validation_and_normalization(self):
        self.assertEqual(self.provider("gemini").name(), "gemini/test-model")
        p = CloudJSONProvider("gemini", "fake", "models/gemini-test")
        self.assertEqual(p.model, "gemini-test")
        for model in ("", "https://evil.invalid", "model?key=secret", "../model"):
            with self.assertRaises(ValueError):
                CloudJSONProvider("gemini", "fake", model)

    def test_check_command_does_not_call_api_by_default(self):
        import check_provider
        config = {"llm_provider": "ollama", "gemini": {"model": "test-model"}}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-secret-never-real"}, clear=True):
            with patch.object(sys, "argv", ["check_provider.py", "--provider", "gemini"]):
                with patch.object(Path, "read_text", return_value=json.dumps(config)), \
                        patch("httpx.post") as post, redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(check_provider.main(), 0)
                    post.assert_not_called()
        self.assertNotIn("fake-secret", output.getvalue())
        self.assertIn("API access is unverified", output.getvalue())

    def test_failed_live_check_does_not_save_configuration(self):
        import check_provider
        config = {"llm_provider": "ollama"}
        with patch.dict(os.environ, {"XAI_API_KEY": "fake-key"}, clear=True):
            with patch.object(sys, "argv", ["check_provider.py", "--provider", "grok", "--live", "--save"]):
                with patch.object(Path, "read_text", return_value=json.dumps(config)):
                    with patch.object(Path, "write_text") as write:
                        with patch("httpx.post", return_value=httpx.Response(401)), redirect_stdout(io.StringIO()):
                            self.assertEqual(check_provider.main(), 1)
                        write.assert_not_called()

    def test_successful_check_saves_selection_but_not_key(self):
        import check_provider
        with patch.dict(os.environ, {"XAI_API_KEY": "fake-secret-never-real"}, clear=True):
            with patch.object(sys, "argv", ["check_provider.py", "--provider", "grok", "--save"]):
                with patch.object(Path, "read_text", return_value='{"llm_provider":"ollama"}'):
                    with patch.object(Path, "write_text") as write, redirect_stdout(io.StringIO()):
                        self.assertEqual(check_provider.main(), 0)
        saved = json.loads(write.call_args.args[0])
        self.assertEqual(saved["llm_provider"], "grok")
        self.assertNotIn("fake-secret", write.call_args.args[0])


@unittest.skipUnless(importlib.util.find_spec("PyQt6"), "PyQt6 not installed")
class SettingsProviderTests(unittest.TestCase):
    def test_round_trip_both_cloud_providers_and_benchmark_budget(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from ui.settings_view import SettingsPanel
        app = QApplication.instance() or QApplication([])
        for provider in ("gemini", "grok", "groq"):
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
                path = Path(handle.name)
            try:
                original = {
                    "llm_provider": provider,
                    provider: {"model": "custom-model", "use_vision": False, "max_tokens": 4096},
                    "agent": {"max_steps_per_task": 80},
                }
                path.write_text(json.dumps(original), encoding="utf-8")
                panel = SettingsPanel(path)
                self.assertEqual(panel.provider_combo.currentData(), provider)
                self.assertEqual(panel.max_steps.value(), 80)
                collected = panel._collect_to_config()
                self.assertEqual(collected["llm_provider"], provider)
                self.assertEqual(collected[provider], original[provider])
                self.assertEqual(collected["agent"]["max_steps_per_task"], 80)
                self.assertNotIn("api_key", collected[provider])
                panel.cloud_models[provider].setText("changed-model")
                panel._collect_to_config()
                self.assertEqual(panel.config, original)
                panel.close()
            finally:
                path.unlink()


if __name__ == "__main__":
    unittest.main()
