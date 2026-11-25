"""Check configuration, or send one tiny API request with --live. No desktop actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.cloud_llm import CLOUD_ORIGINS
from core.llm import build_provider


def positive_seconds(value: str) -> float:
    import math
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("Timeout must be a finite positive number.")
    return seconds


def check_network(provider: str, connect_timeout: float) -> int:
    """Test HTTPS connectivity without sending credentials or inference requests."""
    import httpx
    import os
    import time

    origin = CLOUD_ORIGINS[provider]
    print(f"Checking HTTPS connectivity: {origin}")
    proxy_vars = [
        name for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
                          "https_proxy", "http_proxy", "all_proxy", "no_proxy")
        if os.environ.get(name)
    ]
    print("Proxy variables present (values hidden): " + (", ".join(proxy_vars) or "none"))
    start = time.monotonic()
    try:
        response = httpx.get(
            origin + "/", timeout=httpx.Timeout(30.0, connect=connect_timeout),
            follow_redirects=False,
        )
    except httpx.RequestError as exc:
        print(f"Network check failed: {type(exc).__name__} after {time.monotonic() - start:.1f}s.")
        print("Check connectivity to this host, DNS, and any configured proxy/firewall.")
        return 1
    print(f"HTTPS response received: HTTP {response.status_code} in {time.monotonic() - start:.1f}s.")
    print("A 401/403/404 at the root can be normal. No API key was sent.")
    print("This checks reachability only; credentials, model access and quota remain unverified.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=("gemini", "grok", "groq"))
    parser.add_argument("--model", help="Override the configured API model ID for this check.")
    parser.add_argument("--live", action="store_true", help="Make one billable API request.")
    parser.add_argument("--save", action="store_true", help="Save provider/model after a successful check.")
    parser.add_argument("--connect-timeout", type=positive_seconds,
                        help="Connection establishment timeout in seconds (default 30).")
    parser.add_argument("--network-check", action="store_true",
                        help="Test HTTPS reachability without an API key or inference call.")
    args = parser.parse_args()
    if args.network_check:
        if args.live or args.save:
            parser.error("--network-check cannot be combined with --live or --save")
        return check_network(args.provider, args.connect_timeout or 30)
    path = Path(__file__).with_name("config.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    config["llm_provider"] = args.provider
    if args.model:
        config.setdefault(args.provider, {})["model"] = args.model
    if args.connect_timeout is not None:
        config.setdefault(args.provider, {})["connect_timeout"] = args.connect_timeout
    try:
        provider = build_provider(config)
    except (RuntimeError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return 1
    print(f"Provider: {provider.name()}")
    print("API key is available (value hidden).")
    if args.live:
        response = provider.call_with_schema(
            system_prompt="Return exactly the requested JSON object.",
            user_text='Return {"status":"ok"}.',
            json_schema={
                "type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}},
                "required": ["status"], "additionalProperties": False,
            },
            schema_name="connection_check",
            schema_description="A connection check with status ok.",
        )
        if response.error:
            print(f"API check failed: {response.error}")
            return 1
        print("API check passed: valid structured response received.")
    else:
        print("Configuration check only. API access is unverified; use --live to test.")
    if args.save:
        config.setdefault(args.provider, {})["model"] = provider.model
        path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
        print("Saved provider/model in config.json. No environment keys were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
