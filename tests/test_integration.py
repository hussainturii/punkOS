"""
End-to-end integration test for PunkOS.

Runs the StrategicPlanner with FAKE LLM and FAKE perception so we can verify
the entire pipeline (planner → reasoning → safety → controller → history)
on Linux/CI without needing Ollama, Windows, or a real screen.

What this catches:
- Callback signature mismatches between planner and UI
- Dataclass field changes that break consumers
- Import errors after refactors
- History DB schema bugs
- Safety classifier regressions
- Self-healing loop logic
- Plan-aware execution

What this does NOT catch:
- Real Ollama API behaviour (use tests/test_ollama_smoke.py)
- Real Windows UIA tree walking (test on Windows)
- Real PyAutoGUI clicking (test on Windows)

Run:  python tests/test_integration.py
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub Windows-only modules BEFORE importing core
import types
sys.modules.setdefault("mss", types.ModuleType("mss"))
sys.modules.setdefault("pyautogui", types.ModuleType("pyautogui"))
sys.modules["pyautogui"].FAILSAFE = True
sys.modules["pyautogui"].PAUSE = 0.05
# Make pyautogui calls no-ops so the controller doesn't actually click anything
sys.modules["pyautogui"].click = lambda *a, **kw: None
sys.modules["pyautogui"].doubleClick = lambda *a, **kw: None
sys.modules["pyautogui"].write = lambda *a, **kw: None
sys.modules["pyautogui"].press = lambda *a, **kw: None
sys.modules["pyautogui"].hotkey = lambda *a, **kw: None
sys.modules["pyautogui"].scroll = lambda *a, **kw: None
sys.modules["pyautogui"].moveTo = lambda *a, **kw: None
sys.modules.setdefault("ollama", types.ModuleType("ollama"))
sys.modules["ollama"].Client = lambda *a, **kw: None

from core.llm import LLMProvider, LLMResponse
from core.perception import VisualPerceptionEngine, ScreenState, UIElement
from core.reasoning import ReasoningCore, NEXT_ACTION_SCHEMA, PLAN_SCHEMA
from core.controller import DesktopController
from core.safety import SafetySystem
from core.history import TaskHistoryStore
from core.planner import StrategicPlanner


# ============================================================================
# Fakes
# ============================================================================

class FakeProvider(LLMProvider):
    """A scripted LLM that returns canned responses in order."""

    def __init__(self, scripted_responses: list[dict]):
        self.scripted = list(scripted_responses)
        self.calls = []         # record every call for inspection

    def name(self) -> str:
        return "fake/scripted"

    def supports_vision(self) -> bool:
        return False

    def call_with_schema(self, system_prompt, user_text, json_schema, schema_name,
                         schema_description, image_b64=None):
        self.calls.append({
            "schema_name": schema_name,
            "user_text_excerpt": user_text[:200],
        })
        if not self.scripted:
            return LLMResponse(parsed_json=None, raw_text="", error="ran out of scripted responses")
        next_response = self.scripted.pop(0)
        return LLMResponse(parsed_json=next_response, raw_text=str(next_response))


class FakePerception:
    """A perception engine that returns canned ScreenStates."""

    def __init__(self, states: list[ScreenState]):
        self.states = list(states)
        self.fallback = states[-1] if states else None
        self.captures = 0

    def perceive(self) -> ScreenState:
        self.captures += 1
        if self.states:
            return self.states.pop(0)
        return self.fallback


def make_screen_state(window_title="Untitled - Notepad", elements: Optional[list[UIElement]] = None) -> ScreenState:
    if elements is None:
        elements = [
            UIElement(tag=1, role="ButtonControl", name="OK",
                      bounds=(100, 100, 200, 130), is_enabled=True),
            UIElement(tag=2, role="ButtonControl", name="Cancel",
                      bounds=(220, 100, 320, 130), is_enabled=True),
            UIElement(tag=3, role="EditControl", name="Filename",
                      bounds=(100, 60, 320, 90), is_enabled=True),
        ]
    return ScreenState(
        screenshot_b64="",
        raw_screenshot_b64="",
        elements=elements,
        foreground_window_title=window_title,
        width=1920, height=1080,
    )


# ============================================================================
# Test cases
# ============================================================================

def assert_eq(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def test_happy_path_two_steps():
    """Simulate: plan → click button [1] → done."""
    print("  test_happy_path_two_steps...")

    plan_response = {
        "summary": "Click OK to confirm",
        "feasible": True,
        "steps": [
            {"step_number": 1, "description": "Click the OK button",
             "expected_outcome": "Dialog closes", "is_high_risk": False},
        ],
    }
    click_response = {
        "action_type": "click", "target_tag": 1, "text": "",
        "reasoning": "Click OK", "is_high_risk": False,
    }
    done_response = {
        "action_type": "done", "target_tag": None, "text": "Clicked OK",
        "reasoning": "Goal achieved", "is_high_risk": False,
    }

    provider = FakeProvider([plan_response, click_response, done_response, {"complete": True, "evidence": "OK", "missing": ""}])
    reasoning = ReasoningCore(provider)
    perception = FakePerception([make_screen_state(), make_screen_state(), make_screen_state()])

    db_path = tempfile.mktemp(suffix=".db")
    history = TaskHistoryStore(db_path)

    planner = StrategicPlanner(
        perception=perception,
        reasoning=reasoning,
        controller=DesktopController(),
        safety=SafetySystem(authorized_dirs=["/tmp"], blocked_paths=[]),
        history_store=history,
        max_steps=10,
        step_delay=0.0,
        on_log=lambda _msg: None,    # silence
    )

    result = planner.run("Click the OK button")

    assert_eq(result.success, True, "task success")
    assert_eq(result.steps_taken, 2, "steps taken")    # click + done
    assert result.plan is not None, "plan should exist"
    assert_eq(len(result.plan.steps), 1, "plan step count")
    assert len(provider.calls) == 4, f"expected 4 LLM calls, got {len(provider.calls)}"
    assert_eq(provider.calls[0]["schema_name"], "task_plan", "first call is plan")
    assert_eq(provider.calls[1]["schema_name"], "next_action", "second call is action")

    # History was recorded
    tasks = history.list_tasks()
    assert_eq(len(tasks), 1, "history row count")
    assert_eq(tasks[0]["success"], 1, "history success flag")
    Path(db_path).unlink()
    print("    ✓ pass")


def test_high_risk_blocks_until_confirmed():
    """High-risk action should pause until on_confirm returns True."""
    print("  test_high_risk_blocks_until_confirmed...")

    plan_response = {
        "summary": "Delete the file",
        "feasible": True,
        "steps": [{"step_number": 1, "description": "Click Delete",
                   "expected_outcome": "File gone", "is_high_risk": True}],
    }
    delete_action = {
        "action_type": "click", "target_tag": 1, "text": "",
        "reasoning": "Click delete to remove the file", "is_high_risk": True,
    }
    done_action = {
        "action_type": "done", "target_tag": None, "text": "Deleted",
        "reasoning": "Done", "is_high_risk": False,
    }

    confirm_calls = []

    def fake_confirm(action, assessment):
        confirm_calls.append((action.action_type, assessment.level.value))
        return True   # approve

    provider = FakeProvider([plan_response, delete_action, done_action, {"complete": True, "evidence": "OK", "missing": ""}])
    reasoning = ReasoningCore(provider)
    from unittest.mock import Mock
    from core.window_handoff import Window
    state = make_screen_state()
    state.foreground_window_handle = 42
    perception = FakePerception([state, state, state])
    backend = Mock()
    backend.foreground.return_value = Window(42, state.foreground_window_title, 'notepad.exe')

    db_path = tempfile.mktemp(suffix=".db")
    planner = StrategicPlanner(
        perception=perception,
        reasoning=reasoning,
        controller=DesktopController(backend),
        safety=SafetySystem(authorized_dirs=["/tmp"], blocked_paths=[]),
        history_store=TaskHistoryStore(db_path),
        max_steps=10,
        step_delay=0.0,
        on_log=lambda _: None,
        on_confirm=fake_confirm,
    )

    result = planner.run("Delete file foo.txt")

    assert_eq(result.success, True, "should succeed after approval")
    backend.activate.assert_called_once_with(42)
    assert len(confirm_calls) >= 1, "confirm should have been called at least once"
    assert_eq(confirm_calls[0][1], "HIGH", "first confirm should be HIGH risk")
    Path(db_path).unlink()
    print("    ✓ pass")


def test_high_risk_cancelled():
    """If on_confirm returns False, task should fail with 'User cancelled'."""
    print("  test_high_risk_cancelled...")

    plan_response = {
        "summary": "Delete file",
        "feasible": True,
        "steps": [{"step_number": 1, "description": "Click delete",
                   "expected_outcome": "Done", "is_high_risk": True}],
    }
    delete_action = {
        "action_type": "click", "target_tag": 1, "text": "",
        "reasoning": "Click delete to remove", "is_high_risk": True,
    }

    provider = FakeProvider([plan_response, delete_action])
    perception = FakePerception([make_screen_state(), make_screen_state()])
    db_path = tempfile.mktemp(suffix=".db")

    planner = StrategicPlanner(
        perception=perception,
        reasoning=ReasoningCore(provider),
        controller=DesktopController(),
        safety=SafetySystem(authorized_dirs=["/tmp"], blocked_paths=[]),
        history_store=TaskHistoryStore(db_path),
        max_steps=5,
        step_delay=0.0,
        on_log=lambda _: None,
        on_confirm=lambda _action, _assess: False,    # always cancel
    )

    result = planner.run("Delete file")

    assert_eq(result.success, False, "task should fail when user cancels")
    assert "cancel" in result.final_message.lower(), \
        f"final message should mention cancel: {result.final_message}"
    Path(db_path).unlink()
    print("    ✓ pass")


def test_self_heal_after_failure():
    """If an action fails, planner should call replan_after_failure on the next step."""
    print("  test_self_heal_after_failure...")

    plan = {
        "summary": "Click button",
        "feasible": True,
        "steps": [{"step_number": 1, "description": "Click",
                   "expected_outcome": "Done", "is_high_risk": False}],
    }
    # First action targets a non-existent tag (will fail)
    bad_action = {
        "action_type": "click", "target_tag": 999, "text": "",
        "reasoning": "Click tag 999", "is_high_risk": False,
    }
    # Second action targets a non-existent tag too (will fail)
    bad_action_2 = {
        "action_type": "click", "target_tag": 998, "text": "",
        "reasoning": "Try another tag", "is_high_risk": False,
    }
    # Third action via replan_after_failure: actually click the right one
    recovery_action = {
        "action_type": "click", "target_tag": 1, "text": "",
        "reasoning": "Try tag 1 instead", "is_high_risk": False,
    }
    done = {
        "action_type": "done", "target_tag": None, "text": "OK",
        "reasoning": "done", "is_high_risk": False,
    }

    provider = FakeProvider([plan, bad_action, bad_action_2, recovery_action, done, {"complete": True, "evidence": "OK", "missing": ""}])
    perception = FakePerception([make_screen_state() for _ in range(8)])
    db_path = tempfile.mktemp(suffix=".db")

    planner = StrategicPlanner(
        perception=perception,
        reasoning=ReasoningCore(provider),
        controller=DesktopController(),
        safety=SafetySystem(authorized_dirs=["/tmp"], blocked_paths=[]),
        history_store=TaskHistoryStore(db_path),
        max_steps=10,
        max_consecutive_failures=4,    # allow self-healing room
        step_delay=0.0,
        on_log=lambda _: None,
    )

    result = planner.run("Click the button")

    assert_eq(result.success, True, "should succeed after self-heal")
    Path(db_path).unlink()
    print("    ✓ pass")


def test_callbacks_fire():
    """All planner callbacks (log, status, plan, step, perceive) should fire."""
    print("  test_callbacks_fire...")

    plan = {
        "summary": "Type hello",
        "feasible": True,
        "steps": [{"step_number": 1, "description": "Type",
                   "expected_outcome": "OK", "is_high_risk": False}],
    }
    type_action = {
        "action_type": "type", "target_tag": None, "text": "hello",
        "reasoning": "type", "is_high_risk": False,
    }
    done = {
        "action_type": "done", "target_tag": None, "text": "OK",
        "reasoning": "done", "is_high_risk": False,
    }

    logs, statuses, plans, steps, perceives = [], [], [], [], []

    provider = FakeProvider([plan, type_action, done, {"complete": True, "evidence": "OK", "missing": ""}])
    perception = FakePerception([make_screen_state(), make_screen_state(), make_screen_state()])
    db_path = tempfile.mktemp(suffix=".db")

    planner = StrategicPlanner(
        perception=perception,
        reasoning=ReasoningCore(provider),
        controller=DesktopController(),
        safety=SafetySystem(authorized_dirs=["/tmp"], blocked_paths=[]),
        history_store=TaskHistoryStore(db_path),
        max_steps=5,
        step_delay=0.0,
        on_log=lambda msg: logs.append(msg),
        on_status=lambda s: statuses.append(s),
        on_plan_ready=lambda p: plans.append(p),
        on_step_complete=lambda i, a, r: steps.append((i, a.action_type, r.success)),
        on_perceive=lambda state, target: perceives.append((state.foreground_window_title, target)),
    )

    result = planner.run("Type hello")

    assert_eq(result.success, True, "task success")
    assert len(logs) > 0, "expected log entries"
    assert len(statuses) > 0, "expected status updates"
    assert len(plans) == 1, f"expected 1 plan callback, got {len(plans)}"
    assert len(steps) >= 1, "expected step callbacks"
    assert len(perceives) >= 1, "expected perceive callbacks"
    Path(db_path).unlink()
    print("    ✓ pass")


def test_history_persistence():
    """Tasks should persist across separate TaskHistoryStore instances."""
    print("  test_history_persistence...")

    db_path = tempfile.mktemp(suffix=".db")
    plan = {
        "summary": "ok", "feasible": True,
        "steps": [{"step_number": 1, "description": "x",
                   "expected_outcome": "y", "is_high_risk": False}],
    }
    done = {
        "action_type": "done", "target_tag": None, "text": "OK",
        "reasoning": "done", "is_high_risk": False,
    }

    # Run a task with one store, then read it back with a fresh store
    provider = FakeProvider([plan, done])
    perception = FakePerception([make_screen_state(), make_screen_state()])
    store1 = TaskHistoryStore(db_path)

    planner = StrategicPlanner(
        perception=perception, reasoning=ReasoningCore(provider),
        controller=DesktopController(),
        safety=SafetySystem(authorized_dirs=["/tmp"], blocked_paths=[]),
        history_store=store1, max_steps=5, step_delay=0.0,
        on_log=lambda _: None,
    )
    planner.run("Do something")

    # Fresh store, same DB
    store2 = TaskHistoryStore(db_path)
    tasks = store2.list_tasks()
    assert_eq(len(tasks), 1, "task count after reopen")
    assert "Do something" in tasks[0]["goal"], f"goal: {tasks[0]['goal']}"
    Path(db_path).unlink()
    print("    ✓ pass")


# ============================================================================
# UI-importability test (doesn't run the GUI, just checks imports)
# ============================================================================

def test_ui_modules_importable():
    """Verify all UI modules import cleanly with PyQt6 stubbed."""
    print("  test_ui_modules_importable...")

    # PyQt6 may not be installed in the test env; stub minimally
    if "PyQt6" not in sys.modules:
        pyqt = types.ModuleType("PyQt6")
        sys.modules["PyQt6"] = pyqt
        for sub in ("QtCore", "QtGui", "QtWidgets"):
            mod = types.ModuleType(f"PyQt6.{sub}")
            sys.modules[f"PyQt6.{sub}"] = mod
            setattr(pyqt, sub, mod)

        # pyqtSlot is used as a decorator with arbitrary args, e.g. @pyqtSlot(object, object)
        # So we need a callable that returns a no-op decorator regardless of args.
        def _pyqt_slot(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
        # pyqtSignal can be used like pyqtSignal(str) at class level
        def _pyqt_signal(*args, **kwargs):
            class _Sig:
                def connect(self, *a, **kw): pass
                def emit(self, *a, **kw): pass
            return _Sig()

        sys.modules["PyQt6.QtCore"].pyqtSlot = _pyqt_slot
        sys.modules["PyQt6.QtCore"].pyqtSignal = _pyqt_signal

        for name in ("Qt", "QObject", "QThread", "QTimer", "QUrl", "QMetaObject",
                     "QPropertyAnimation", "QEasingCurve", "QSize"):
            setattr(sys.modules["PyQt6.QtCore"], name, type(name, (), {}))
        sys.modules["PyQt6.QtCore"].pyqtProperty = lambda *a, **kw: (lambda fn: fn)
        for name in (
            "QFont", "QTextCursor", "QPalette", "QColor", "QImage", "QPixmap",
            "QDesktopServices", "QPainter", "QPen", "QKeySequence", "QShortcut",
        ):
            setattr(sys.modules["PyQt6.QtGui"], name, type(name, (), {}))
        widget_names = (
            "QApplication", "QMainWindow", "QWidget", "QVBoxLayout", "QHBoxLayout",
            "QPushButton", "QTextEdit", "QLineEdit", "QLabel", "QListWidget",
            "QListWidgetItem", "QFrame", "QSplitter", "QMessageBox", "QDialog",
            "QDialogButtonBox", "QCheckBox", "QStatusBar", "QSizePolicy",
            "QStackedWidget", "QTabWidget", "QProgressBar", "QComboBox",
            "QFileDialog", "QSpinBox", "QFormLayout", "QGroupBox",
            "QTreeWidget", "QTreeWidgetItem", "QPlainTextEdit",
            "QTableWidget", "QTableWidgetItem", "QHeaderView",
            "QScrollArea", "QGraphicsOpacityEffect",
        )
        for name in widget_names:
            setattr(sys.modules["PyQt6.QtWidgets"], name, type(name, (), {}))

    # Now try the imports
    import importlib
    for mod_name in (
        "ui.execution_view",
        "ui.file_search_view",
        "ui.history_view",
        "ui.settings_view",
        "ui.plugin_manager",
        "ui.web_agent_view",
    ):
        try:
            importlib.import_module(mod_name)
            # Don't try ui.dashboard here — it imports too many concrete Qt classes
            # that our minimal stub can't satisfy. The syntax check is enough.
        except Exception as e:
            raise AssertionError(f"{mod_name} failed to import: {e}")
    print("    ✓ pass")


# ============================================================================
# Driver
# ============================================================================

def main():
    print("─" * 60)
    print("PunkOS Integration Tests")
    print("─" * 60)
    tests = [
        test_happy_path_two_steps,
        test_high_risk_blocks_until_confirmed,
        test_high_risk_cancelled,
        test_self_heal_after_failure,
        test_callbacks_fire,
        test_history_persistence,
        test_ui_modules_importable,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"    ✗ FAIL: {e}")
            failed.append((t.__name__, e))

    print()
    print("─" * 60)
    if failed:
        print(f"❌ {len(failed)}/{len(tests)} tests failed:")
        for name, e in failed:
            print(f"   - {name}: {e}")
        sys.exit(1)
    print(f"✅ All {len(tests)} tests passed.")
    print("─" * 60)


if __name__ == "__main__":
    main()
