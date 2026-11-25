"""Gemini, Grok and Groq JSON providers using their documented REST endpoints.

Keys come from the environment via build_provider(), never from saved settings.
Transient server failures receive one bounded retry, without replaying UI actions.
"""
from __future__ import annotations

import copy
import json
import re
import time
from typing import Optional
from urllib.parse import quote

from .llm import LLMProvider, LLMResponse
from .inference_transport import post_response, InferenceCancelled

DEFAULT_CLOUD_MODELS = {
    "gemini": "gemini-3.8-flash",
    "grok": "grok-4.7",
    "groq": "qwen/qwen3.8-27b",
}
CLOUD_ORIGINS = {
    "gemini": "https://generativelanguage.googleapis.com",
    "grok": "https://api.x.ai",
    "groq": "https://api.groq.com",
}


def closed_schema(schema: dict) -> dict:
    """Close object schemas without changing the caller's shared schema."""
    result = copy.deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


class CloudJSONProvider(LLMProvider):
    def __init__(
        self, provider: str, api_key: str, model: str, max_tokens: int = 8192,
        use_vision: bool = False, temperature: Optional[float] = None,
        request_timeout: int = 180, connect_timeout: float = 30,
    ):
        try:
            import httpx
            import jsonschema
        except ImportError as exc:
            raise RuntimeError("Cloud providers require: python -m pip install httpx jsonschema") from exc
        if provider not in DEFAULT_CLOUD_MODELS:
            raise ValueError("Cloud provider must be gemini, grok or groq")
        model = model.strip()
        if provider == "gemini":
            model = model.removeprefix("models/")
        model_pattern = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
        if provider == "groq":
            model_pattern += r"(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?"
        if not re.fullmatch(model_pattern, model):
            raise ValueError("Enter a valid API model ID, not a URL or display name")
        if max_tokens <= 0 or request_timeout <= 0 or connect_timeout <= 0:
            raise ValueError("max_tokens, request_timeout and connect_timeout must be positive")
        self.provider = provider
        self._api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.use_vision = use_vision
        self.temperature = temperature
        self.cancel_check = None
        self.on_retry = None
        self._httpx = httpx
        self._jsonschema = jsonschema
        self._timeout = httpx.Timeout(
            connect=float(connect_timeout), read=float(request_timeout), write=60.0, pool=5.0,
        )

    def name(self) -> str:
        return f"{self.provider}/{self.model}"

    def supports_vision(self) -> bool:
        return self.use_vision

    def _error_detail(self, response) -> str:
        """Extract diagnostic fields, never dump response bodies or headers."""
        try:
            data = response.json()
        except ValueError:
            return ""
        if not isinstance(data, dict):
            return ""
        error = data.get("error")
        fields = []
        if isinstance(error, dict):
            # OpenAI-compatible and Gemini error envelopes.
            for name in ("message", "code", "type", "param", "status"):
                value = error.get(name)
                if isinstance(value, (str, int)) and not isinstance(value, bool):
                    fields.append(f"{name}: {value}")
        else:
            # xAI also returns {"code": "...", "error": "human-readable reason"}.
            if isinstance(error, str):
                fields.append(error)
            for name in ("message", "code"):
                value = data.get(name)
                if isinstance(value, (str, int)) and not isinstance(value, bool):
                    fields.append(f"{name}: {value}")
        detail = " | ".join(fields)
        # Redact before truncation so a key crossing the length limit cannot leak.
        if self._api_key:
            for secret in (self._api_key, quote(self._api_key, safe="")):
                detail = detail.replace(secret, "[REDACTED]")
        detail = re.sub(r"(?i)\bBearer\s+[^\s,;\"']+", "Bearer [REDACTED]", detail)
        detail = re.sub(
            r"(?i)\b(?:xai-|sk-|gsk_|AIza)[A-Za-z0-9_-]{8,}", "[REDACTED]", detail,
        )
        # Error text is external data: keep it on one line without terminal escapes.
        detail = "".join(c if c.isprintable() else " " for c in detail)
        detail = " ".join(detail.split())
        return detail[:600] + ("..." if len(detail) > 600 else "")

    def call_with_schema(
        self, system_prompt: str, user_text: str, json_schema: dict,
        schema_name: str, schema_description: str, image_b64: Optional[str] = None,
    ) -> LLMResponse:
        schema = closed_schema(json_schema)
        system = f"{system_prompt}\n\nReturn the requested JSON object: {schema_description}"
        image = image_b64 if self.use_vision else None
        if self.provider == "gemini":
            url = (
                f"{CLOUD_ORIGINS['gemini']}/v1beta/models/"
                f"{self.model}:generateContent"
            )
            headers = {"x-goog-api-key": self._api_key}
            parts = [{"text": user_text}]
            if image:
                parts.append({"inlineData": {"mimeType": "image/png", "data": image}})
            generation = {
                "maxOutputTokens": self.max_tokens,
                "responseFormat": {"text": {"mimeType": "APPLICATION_JSON", "schema": schema}},
            }
            if self.temperature is not None:
                generation["temperature"] = self.temperature
            payload = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": generation,
            }
        else:
            url = (
                f"{CLOUD_ORIGINS['groq']}/openai/v1/chat/completions"
                if self.provider == "groq" else f"{CLOUD_ORIGINS['grok']}/v1/chat/completions"
            )
            headers = {"Authorization": f"Bearer {self._api_key}"}
            content = user_text
            if image:
                content = [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                ]
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                ("max_completion_tokens" if self.provider == "groq" else "max_tokens"): self.max_tokens,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
            }
            if self.temperature is not None:
                payload["temperature"] = self.temperature
            if self.provider == "groq" and self.model.startswith("qwen/"):
                # Keep reasoning outside the JSON content consumed by the executor.
                payload["reasoning_format"] = "hidden"

        try:
            for attempt in range(2):
                if self.cancel_check is not None:
                    response = post_response(url, payload, headers, self._timeout, self.cancel_check)
                else:
                    response = self._httpx.post(
                        url, headers=headers, json=payload, timeout=self._timeout,
                        follow_redirects=False,
                    )
                if response.status_code not in {502, 503, 504} or attempt:
                    break
                delay = 2.0
                try:
                    delay = float(response.headers.get('Retry-After', '2'))
                except (ValueError, TypeError):
                    pass
                # Do not retry earlier than a long server-requested delay.
                if not 0 <= delay <= 10:
                    break
                message = f'{self.provider} HTTP {response.status_code}; retrying inference once in {delay:g}s (no desktop action replay).'
                if callable(self.on_retry):
                    self.on_retry(message)
                else:
                    print(message)
                for _ in range(max(1, int(delay * 10))):
                    if self.cancel_check is not None and self.cancel_check():
                        raise InferenceCancelled()
                    time.sleep(delay / max(1, int(delay * 10)))
        except InferenceCancelled:
            return LLMResponse(None, '', 'Cloud inference cancelled.')
        except self._httpx.RequestError as exc:
            # Do not include exception text: transport errors may contain secrets.
            message = f"{self.provider} request failed ({type(exc).__name__})."
            if isinstance(exc, self._httpx.ConnectTimeout):
                message += (
                    f" Connection establishment exceeded {self._timeout.connect:g}s."
                    " No API response was received; key/model validity is unknown."
                    f" Run: python check_provider.py --provider {self.provider} --network-check"
                )
            elif isinstance(exc, self._httpx.ReadTimeout):
                message += " Timed out waiting for response data. Check network conditions or request_timeout."
            return LLMResponse(None, "", message)
        if response.status_code != 200:
            hints = {
                400: "Check model settings and structured-output compatibility.",
                401: "Check the API key in this terminal session.",
                403: "Check API key permissions and model access.",
                404: "Check the exact model ID and its availability for your account.",
                429: "Check provider quota, billing and rate limits; retry later.",
                503: "Provider temporarily unavailable. This response does not establish a quota or API-key problem.",
            }
            hint = hints.get(response.status_code, "Check provider availability and your account.")
            detail = self._error_detail(response)
            message = f"{self.provider} HTTP {response.status_code} (model {self.model})."
            if detail:
                message += f" Provider error: {detail}"
            return LLMResponse(None, "", f"{message} {hint}")

        raw = ""
        try:
            data = response.json()
            if self.provider == "gemini":
                candidates = data.get("candidates", [])
                if not candidates:
                    return LLMResponse(None, "", "gemini returned no candidate (possibly blocked).")
                candidate = candidates[0]
                if candidate.get("finishReason") != "STOP":
                    return LLMResponse(None, "", "gemini did not finish normally; output may be blocked or truncated.")
                raw = "".join(
                    part.get("text", "") for part in candidate.get("content", {}).get("parts", [])
                    if not part.get("thought", False)
                )
            else:
                choice = data["choices"][0]
                if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
                    return LLMResponse(None, "", f"{self.provider} did not finish normally; output may be refused or truncated.")
                raw = choice["message"]["content"]
            # Verification copies UI text, including table tabs/newlines. Some
            # responses leave these literal inside JSON strings. Preserve the
            # exact text for quote validation; never relax action JSON parsing.
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                if (schema_name != 'verify_completion'
                        or not exc.msg.startswith('Invalid control character')
                        or any(ord(c) < 32 and c not in '\t\r\n' for c in raw)):
                    raise
                parsed = json.loads(raw, strict=False)
            if not isinstance(parsed, dict):
                return LLMResponse(None, raw, f"{self.provider} returned JSON that is not an object.")
            self._jsonschema.validate(parsed, schema)
        except self._jsonschema.ValidationError as exc:
            # Report only the schema rule, never model values or arbitrary
            # property names (which may contain screen content or secrets).
            rule = str(exc.validator) if exc.validator in {
                "type", "required", "enum", "additionalProperties", "minimum",
                "maximum", "minItems", "maxItems", "minLength", "maxLength",
                "anyOf", "oneOf", "pattern",
            } else "validation"
            return LLMResponse(
                None, raw,
                f"{self.provider} output did not match the requested schema (rule: {rule}). "
                "No action from this response was executed.",
            )
        except self._jsonschema.SchemaError:
            return LLMResponse(None, raw, "The application supplied an invalid JSON schema.")
        except json.JSONDecodeError as exc:
            return LLMResponse(
                None, raw if isinstance(raw, str) else '',
                f'{self.provider} returned malformed JSON at line {exc.lineno}, column {exc.colno}.')
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            return LLMResponse(None, raw if isinstance(raw, str) else "", f"{self.provider} returned an invalid response.")
        return LLMResponse(parsed, raw)
