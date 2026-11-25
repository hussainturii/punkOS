"""
Test the perception layer in isolation.
Run while a Windows app (e.g. Calculator, Notepad) has focus, then open
perception_debug.png to see what the LLM will see.

    python tests/test_perception.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.perception import VisualPerceptionEngine, UIA_AVAILABLE


def main():
    print(f"UIA available: {UIA_AVAILABLE}")
    engine = VisualPerceptionEngine()
    state = engine.perceive()
    print(f"Foreground window: {state.foreground_window_title}")
    print(f"Screen size: {state.width} x {state.height}")
    print(f"Found {len(state.elements)} interactive elements:")
    for el in state.elements[:25]:
        print(f"  [{el.tag:2}] {el.role:20} '{el.name}'  enabled={el.is_enabled}  bounds={el.bounds}")
    if len(state.elements) > 25:
        print(f"  ... and {len(state.elements) - 25} more")

    # Save the annotated screenshot
    annotated = engine.annotate_screenshot(engine.capture_screen(), state.elements)
    out = Path(__file__).parent.parent / "perception_debug.png"
    annotated.save(out)
    print(f"\nSaved annotated screenshot to {out}")
    print("Open it to see exactly what the LLM will see when it makes decisions.")


if __name__ == "__main__":
    main()
