"""
LLM provider abstraction.

Architecture decision for 8GB RAM / no GPU systems:
We default to TEXT-ONLY mode, sending only the accessibility tree text + goal
to a small local model (qwen2.5:3b, ~2GB RAM). This is the right choice because:

1. Vision-language models that work for UI tasks need 6GB+ RAM and 30-90s/step on CPU.
2. The Windows accessibility tree can expose controls without image inference,
   but some applications provide incomplete trees and need another controller.
3. CPU latency varies substantially with prompt length and hardware. Measure it;
   a small model alone does not guarantee fast or reliable task completion.

Vision is wired up as an optional capability for when accessibility tree is empty
(rare on Windows; common in Electron/Chrome canvas apps), but defaults to off.

Ollama, Anthropic, Gemini, Grok and Groq share the same interface. Select the backend
using config.json["llm_provider"]; Gemini, Grok and Groq keys come from the environment.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

try:
    import httpx as _httpx
except ImportError:
    _httpx = None  # type: ignore


@dataclass
class LLMResponse:
    """Normalised response across providers."""
    parsed_json: Optional[dict]   # Structured output if we got one
    raw_text: str                 # Full text response, for debugging
    error: Optional[str] = None


class LLMProvider(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def call_with_schema(
        self,
        system_prompt: str,
        user_text: str,
        json_schema: dict,
        schema_name: str,
        schema_description: str,
        image_b64: Optional[str] = None,
    ) -> LLMResponse:
        """Send a prompt and force structured JSON output. Image is optional."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def supports_vision(self) -> bool:
        ...


# ---------------------------------------------------------------------------
# Ollama provider (PRIMARY — local)
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Talks to a local Ollama instance using the official Python client."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        vision_model: Optional[str] = None,
        use_vision: bool = False,
        temperature: float = 0.1,
        num_ctx: int = 4096,
        request_timeout: int = 180,
        num_predict: int = 512,
        keep_alive: str = '10m',
        think: Optional[bool] = None,
    ):
        try:
            import ollama
        except ImportError as e:
            raise RuntimeError(
                "ollama package missing. Install with: pip install ollama"
            ) from e

        # Use httpx.Timeout so the long read-timeout applies to LLM inference,
        # separately from connection establishment.
        if _httpx is not None:
            _timeout = _httpx.Timeout(connect=10.0, read=float(request_timeout), write=60.0, pool=5.0)
        else:
            _timeout = request_timeout
        self.client = ollama.Client(host=host, timeout=_timeout)
        self.host = host.rstrip('/')
        if '://' not in self.host:
            self.host = 'http://' + self.host
        self.request_timeout = float(request_timeout)
        self.cancel_check = lambda: False
        self.text_model = model
        self.vision_model = vision_model or model
        self.use_vision = use_vision
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.think = think
        self.last_metrics = {}
        self._verify_models()

    def _verify_models(self):
        """Check requested model is pulled. Print clear instructions if not."""
        try:
            installed = self.client.list()
            installed_names = set()
            for m in installed.models:
                installed_names.add(m.model)
                # Also accept the prefix without tag
                installed_names.add(m.model.split(":")[0])
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama. Is it running?\n"
                f"  Start it with:  ollama serve\n"
                f"  Or check the host in config.json\n"
                f"Underlying error: {e}"
            )

        # Check the text model
        if not self._model_present(self.text_model, installed_names):
            raise RuntimeError(
                f"\nModel '{self.text_model}' not found in Ollama.\n"
                f"Pull it with:\n"
                f"    ollama pull {self.text_model}\n\n"
                f"Currently installed: {sorted(n for n in installed_names if ':' in n)}"
            )

        # Vision model only required if use_vision=true
        if self.use_vision and not self._model_present(self.vision_model, installed_names):
            raise RuntimeError(
                f"\nVision model '{self.vision_model}' not found in Ollama (and use_vision=true).\n"
                f"Pull it with:\n"
                f"    ollama pull {self.vision_model}\n"
                f"OR set 'use_vision': false in config.json to skip vision."
            )

    def ensure_available(self):
        """Bounded, cancellable readiness check before a task touches the UI."""
        from .inference_transport import get_json
        data = get_json(self.host + '/api/tags', 3.0, self.cancel_check)
        names = {entry.get('model') or entry.get('name', '') for entry in data.get('models', [])}
        needed = [self.text_model] + ([self.vision_model] if self.use_vision else [])
        for name in needed:
            if not self._model_present(name, names):
                raise RuntimeError(f'Configured Ollama model is not installed: {name}')

    @staticmethod
    def _model_present(needed: str, installed_names: set[str]) -> bool:
        if needed in installed_names:
            return True
        if ':' in needed:
            return False
        return any(name.startswith(needed + ":") for name in installed_names)

    def name(self) -> str:
        if self.use_vision:
            return f"ollama/{self.text_model}+{self.vision_model}"
        return f"ollama/{self.text_model}"

    def supports_vision(self) -> bool:
        return self.use_vision

    def call_with_schema(
        self,
        system_prompt: str,
        user_text: str,
        json_schema: dict,
        schema_name: str,
        schema_description: str,
        image_b64: Optional[str] = None,
    ) -> LLMResponse:
        # Pick model + decide whether to send the image
        self.last_metrics = {}
        send_image = self.use_vision and image_b64 is not None
        model_to_use = self.vision_model if send_image else self.text_model

        # Augment system prompt with schema reminder
        full_system = (
            f"{system_prompt}\n\n"
            f"Return one JSON object for '{schema_name}'. "
            f"{schema_description}\n\n"
            f"Return ONLY the JSON object. No markdown fences. No commentary. No explanation."
        )

        user_message: dict[str, Any] = {"role": "user", "content": user_text}
        if send_image:
            user_message["images"] = [image_b64]

        messages = [
            {"role": "system", "content": full_system},
            user_message,
        ]

        try:
            from .inference_transport import post_json
            response = post_json(self.host + '/api/chat', {
                'model': model_to_use,
                'messages': messages,
                'format': json_schema,
                'options': {
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
                'keep_alive': self.keep_alive,
                **({'think': self.think} if self.think is not None else {}),
                'stream': False,
            }, self.request_timeout, self.cancel_check)
        except Exception as e:
            return LLMResponse(parsed_json=None, raw_text="", error=f"Ollama call failed: {e}")

        if hasattr(response, "message"):
            raw = response.message.content
        else:
            raw = response.get("message", {}).get("content", "")

        self.last_metrics = {k: getattr(response, k, None) if not isinstance(response, dict) else response.get(k)
                             for k in ('total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration')}
        parsed = self._extract_json(raw)
        if parsed is None:
            return LLMResponse(
                parsed_json=None,
                raw_text=raw,
                error=f"Could not extract valid JSON from response. Got: {raw[:200]}"
            )
        try:
            import jsonschema
            jsonschema.validate(parsed, json_schema)
        except (jsonschema.ValidationError, jsonschema.SchemaError):
            return LLMResponse(None, raw, 'Ollama output did not match the action schema.')
        return LLMResponse(parsed_json=parsed, raw_text=raw)

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Robustly pull a JSON object out of a response."""
        if not text:
            return None
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # ```json fences
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # First { to last }
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None


# ---------------------------------------------------------------------------
# Anthropic provider (FALLBACK — cloud)
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Talks to the Claude API. Forced tool use guarantees JSON output."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5", max_tokens: int = 2048):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package missing. Install with: pip install anthropic"
            ) from e
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def name(self) -> str:
        return f"anthropic/{self.model}"

    def supports_vision(self) -> bool:
        return True

    def call_with_schema(
        self,
        system_prompt: str,
        user_text: str,
        json_schema: dict,
        schema_name: str,
        schema_description: str,
        image_b64: Optional[str] = None,
    ) -> LLMResponse:
        tool = {
            "name": schema_name,
            "description": schema_description,
            "input_schema": json_schema,
        }

        content: list[dict] = []
        if image_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
            })
        content.append({"type": "text", "text": user_text})

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                tools=[tool],
                tool_choice={"type": "tool", "name": schema_name},
                messages=[{"role": "user", "content": content}],
            )
        except Exception as e:
            return LLMResponse(parsed_json=None, raw_text="", error=f"Anthropic call failed: {e}")

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if not tool_use:
            return LLMResponse(parsed_json=None,
                               raw_text=str(response.content),
                               error="No tool_use block in response")
        return LLMResponse(parsed_json=dict(tool_use.input), raw_text=str(tool_use.input))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_provider(config: dict) -> LLMProvider:
    """Construct the configured provider from config.json."""
    provider_name = config.get("llm_provider", "ollama").lower()

    if provider_name == "ollama":
        oc = config.get("ollama", {})
        return OllamaProvider(
            host=oc.get("host", "http://localhost:11434"),
            model=oc.get("model", "qwen2.5:3b"),
            vision_model=oc.get("vision_model", "moondream:latest"),
            use_vision=oc.get("use_vision", False),
            temperature=oc.get("temperature", 0.1),
            num_ctx=oc.get("num_ctx", 4096),
            request_timeout=oc.get("request_timeout", 180),
            num_predict=oc.get('num_predict', 512),
            keep_alive=oc.get('keep_alive', '10m'),
            think=oc.get('think'),
        )

    if provider_name == "anthropic":
        import os
        ac = config.get("anthropic", {})
        api_key = os.environ.get("ANTHROPIC_API_KEY") or ac.get("api_key")
        if not api_key or api_key.startswith("PUT_"):
            raise RuntimeError("Set ANTHROPIC_API_KEY env var or fill anthropic.api_key in config.json")
        return AnthropicProvider(
            api_key=api_key,
            model=ac.get("model", "claude-sonnet-4-5"),
            max_tokens=ac.get("max_tokens", 2048),
        )

    if provider_name in ("gemini", "grok", "groq"):
        import os
        from .cloud_llm import CloudJSONProvider, DEFAULT_CLOUD_MODELS

        key_name = {
            "gemini": "GEMINI_API_KEY", "grok": "XAI_API_KEY", "groq": "GROQ_API_KEY",
        }[provider_name]
        api_key = os.environ.get(key_name, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Set {key_name} in PowerShell, then launch PunkOS from that same "
                "terminal. Already-running apps do not inherit new variables."
            )
        cloud = config.get(provider_name, {})
        instance = CloudJSONProvider(
            provider=provider_name,
            api_key=api_key,
            model=cloud.get("model", DEFAULT_CLOUD_MODELS[provider_name]),
            max_tokens=cloud.get("max_tokens", 4096 if provider_name == "groq" else 8192),
            use_vision=cloud.get("use_vision", False),
            temperature=cloud.get("temperature"),
            request_timeout=cloud.get("request_timeout", 180),
            connect_timeout=cloud.get("connect_timeout", 30),
        )
        instance.vision_for_actions = cloud.get('vision_for_actions', True)
        return instance

    raise ValueError(
        f"Unknown llm_provider: {provider_name!r}. "
        "Use 'ollama', 'anthropic', 'gemini', 'grok', or 'groq'."
    )
