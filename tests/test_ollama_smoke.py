"""
Smoke test for the Ollama setup.

Run BEFORE you launch PunkOS:
    python tests/test_ollama_smoke.py

It does:
  1. Confirm Ollama is reachable on localhost:11434
  2. Confirm your configured text model is pulled
  3. Send a small prompt and verify it returns valid structured JSON
  4. (If use_vision=true) test the vision model too
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm import OllamaProvider


def main():
    config_path = Path(__file__).parent.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    oc = config["ollama"]

    print("─" * 60)
    print("PunkOS Ollama Smoke Test")
    print("─" * 60)
    print(f"Host:        {oc['host']}")
    print(f"Text model:  {oc['model']}")
    print(f"Use vision:  {oc.get('use_vision', False)}")
    if oc.get("use_vision"):
        print(f"Vision model: {oc.get('vision_model')}")
    print()

    # Step 1
    print("[1/3] Connecting to Ollama and verifying model is pulled...")
    try:
        provider = OllamaProvider(
            host=oc["host"],
            model=oc["model"],
            vision_model=oc.get("vision_model"),
            use_vision=oc.get("use_vision", False),
            temperature=oc.get("temperature", 0.1),
            num_ctx=oc.get("num_ctx", 4096),
            request_timeout=oc.get("request_timeout", 180),
        )
        print(f"      ✓ Connected. Provider: {provider.name()}")
    except Exception as e:
        print(f"      ❌ {e}")
        sys.exit(1)

    # Step 2: text-only structured output (this is what PunkOS actually uses)
    print("\n[2/3] Sending text-only structured-output prompt...")
    schema = {
        "type": "object",
        "properties": {
            "action_type": {
                "type": "string",
                "enum": ["click", "type", "done"],
            },
            "target_tag": {"type": ["integer", "null"]},
            "reasoning": {"type": "string"},
        },
        "required": ["action_type", "reasoning", "target_tag"],
    }
    response = provider.call_with_schema(
        system_prompt=(
            "You are PunkOS, a test agent. Pretend the user wants you to click "
            "the OK button which has tag [3]. Return the appropriate action."
        ),
        user_text=(
            "GOAL: Click the OK button.\n\n"
            "ELEMENTS:\n"
            "  [1] ButtonControl: Cancel\n"
            "  [2] EditControl: filename\n"
            "  [3] ButtonControl: OK\n\n"
            "Decide the next action."
        ),
        json_schema=schema,
        schema_name="next_action",
        schema_description="A test next-action decision.",
    )
    if response.error:
        print(f"      ❌ {response.error}")
        sys.exit(1)
    print(f"      ✓ Got JSON: {response.parsed_json}")
    if response.parsed_json.get("target_tag") == 3:
        print("      ✓ Model picked the correct tag (3 = OK button)")
    else:
        print(f"      ⚠ Model picked tag {response.parsed_json.get('target_tag')} instead of 3 — "
              "this small model may struggle. Consider qwen2.5:7b if you have RAM.")

    # Step 3: optional vision
    if provider.supports_vision():
        print("\n[3/3] Vision enabled — sending tiny test image...")
        from PIL import Image, ImageDraw
        import io, base64
        img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 80, 80], fill=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        v_schema = {
            "type": "object",
            "properties": {"primary_color": {"type": "string"}},
            "required": ["primary_color"],
        }
        response = provider.call_with_schema(
            system_prompt="Describe what you see in the image.",
            user_text="What is the primary color of the shape in this image?",
            json_schema=v_schema,
            schema_name="color_query",
            schema_description="Identify dominant color.",
            image_b64=b64,
        )
        if response.error:
            print(f"      ❌ {response.error}")
            sys.exit(1)
        print(f"      ✓ Got JSON: {response.parsed_json}")
    else:
        print("\n[3/3] Vision disabled in config — skipping vision test.")

    print("\n" + "─" * 60)
    print("✅ Ollama setup looks good. PunkOS is ready to run.")
    print("─" * 60)
    print("\nNext steps:")
    print("  • Test perception:  python tests/test_perception.py")
    print("  • Run a CLI task:   python run_cli.py \"Open Calculator\"")
    print("  • Launch the GUI:   python main.py")


if __name__ == "__main__":
    main()
