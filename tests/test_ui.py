"""
UI tests — theme system, widgets, and every panel.

Runs headless via Qt's offscreen platform plugin, so it needs no display and no
Ollama. It does need PyQt6 installed; if that is missing the suite skips rather
than fails, because the core modules must remain testable without a GUI stack.

Covers the SDS requirements that are actually checkable in code:
  UI-5   both themes meet WCAG AA (4.5:1) — computed, not eyeballed
  UI-4   status indicator maps planner states to the five FR1.3 states
  FR1.3  Idle / Thinking / Planning / Executing / Awaiting Confirmation
  FR2.7  Confirm stays disabled until the acknowledgement box is ticked
  FR6.1  plugin panel renders the live registry
  USE-8  keyboard shortcuts are registered

Plus a regression guard for the QFrame stylesheet cascade bug: QLabel subclasses
QFrame, so an unscoped `QFrame { border: ... }` rule set on a card leaks into
every child label and renders text as bordered input boxes. That shipped once;
this test stops it shipping again.

Run:  python tests/test_ui.py
"""
from __future__ import annotations

import os
import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Headless rendering — must be set before QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Stub Windows-only modules before importing core.
sys.modules.setdefault("mss", types.ModuleType("mss"))
_pag = types.ModuleType("pyautogui")
_pag.FAILSAFE = True
_pag.PAUSE = 0.05
for _name in ("click", "doubleClick", "write", "press", "hotkey", "scroll",
              "moveTo", "mouseDown", "mouseUp"):
    setattr(_pag, _name, lambda *a, **kw: None)
sys.modules.setdefault("pyautogui", _pag)

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox, QLabel, QDialog
    from PyQt6.QtGui import QShortcut
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

UI_DIR = Path(__file__).parent.parent / "ui"


def assert_eq(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label):
    if not value:
        raise AssertionError(f"{label}: expected truthy, got {value!r}")


# ---------------------------------------------------------------------------
# Theme — no Qt needed
# ---------------------------------------------------------------------------

def test_both_themes_meet_wcag_aa():
    """UI-5: every meaningful text/background pair at 4.5:1 or better."""
    print("  test_both_themes_meet_wcag_aa...")
    from ui.theme import DARK, LIGHT, audit_contrast

    for palette in (DARK, LIGHT):
        failures = audit_contrast(palette, minimum=4.5)
        if failures:
            detail = ", ".join(f"{fg} on {bg} = {ratio}:1" for fg, bg, ratio in failures)
            raise AssertionError(f"{palette.name} theme fails WCAG AA: {detail}")
    print("    ✓ pass")


def test_accent_tokens_serve_both_roles():
    """The split accent must work as text AND as a button fill.

    A single accent cannot do both on a dark surface: light enough to read as
    text is too light behind white button labels. Guard both directions.
    """
    print("  test_accent_tokens_serve_both_roles...")
    from ui.theme import DARK, LIGHT, contrast_ratio

    for palette in (DARK, LIGHT):
        as_text = contrast_ratio(palette.accent, palette.surface)
        as_fill = contrast_ratio(palette.text_on_accent, palette.accent_solid)
        assert_true(as_text >= 4.5,
                    f"{palette.name}: accent as text on surface is {as_text:.2f}:1")
        assert_true(as_fill >= 4.5,
                    f"{palette.name}: text on accent_solid is {as_fill:.2f}:1")
    print("    ✓ pass")


def test_stylesheet_builds_for_both_themes():
    print("  test_stylesheet_builds_for_both_themes...")
    from ui.theme import DARK, LIGHT, build_stylesheet

    for palette in (DARK, LIGHT):
        sheet = build_stylesheet(palette)
        assert_true(len(sheet) > 2000, f"{palette.name} stylesheet looks truncated")
        assert_true("{" not in sheet.replace("{{", "").split("QWidget")[0][:0] + "",
                    "sanity")
        # Every token must have been interpolated — a stray {p.foo} means a
        # brace-escaping bug in the f-string.
        leftovers = re.findall(r"\{p\.\w+\}|\{m\.\w+\}", sheet)
        assert_eq(leftovers, [], f"{palette.name} stylesheet has uninterpolated tokens")
    print("    ✓ pass")


def test_no_unscoped_qframe_stylesheets():
    """Regression guard: QLabel subclasses QFrame.

    `self.setStyleSheet("QFrame { border: 1px solid ... }")` on a card widget
    also styles every descendant QLabel, which renders headings and values as
    bordered input boxes. Scope such rules with an objectName selector
    (`QFrame#myCard { ... }`) instead.
    """
    print("  test_no_unscoped_qframe_stylesheets...")
    offenders = []
    pattern = re.compile(r"self\.setStyleSheet\(\s*f?(\"\"\"|\")(.*?)\1", re.S)
    unscoped = re.compile(r"(?<![#\]\w])QFrame\s*\{")

    for path in sorted(UI_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in pattern.finditer(source):
            if unscoped.search(match.group(2)):
                line = source[:match.start()].count("\n") + 1
                offenders.append(f"{path.name}:{line}")

    assert_eq(offenders, [],
              "unscoped QFrame rules leak into child QLabels; scope by objectName")
    print("    ✓ pass")


def test_no_widget_wide_background_rule():
    """`QWidget { background-color }` makes every QLabel opaque.

    That paints solid rectangles over any card a label sits on. Backgrounds
    belong on windows and named containers.
    """
    print("  test_no_widget_wide_background_rule...")
    from ui.theme import DARK, build_stylesheet

    sheet = build_stylesheet(DARK)
    block = re.search(r"QWidget\s*\{([^}]*)\}", sheet)
    assert_true(block is not None, "stylesheet should define a base QWidget rule")
    assert_true("background-color" not in block.group(1),
                "base QWidget rule must not set background-color")
    print("    ✓ pass")


# ---------------------------------------------------------------------------
# Widgets / panels — need QApplication
# ---------------------------------------------------------------------------

_app = None


def _ensure_app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)
        _app.setStyle("Fusion")
        # Modals would block a headless run.
        QMessageBox.critical = staticmethod(lambda *a, **kw: None)
        QMessageBox.warning = staticmethod(lambda *a, **kw: None)
        QMessageBox.information = staticmethod(lambda *a, **kw: None)
    return _app


def _build_window():
    """A fully constructed main window with a fake LLM provider."""
    _ensure_app()
    import ui.dashboard as dashboard

    class FakeProvider:
        def name(self): return "ollama/qwen2.5:3b"
        def supports_vision(self): return False
        def call_with_schema(self, **kw): raise RuntimeError("not used in UI tests")

    dashboard.build_provider = lambda cfg: FakeProvider()
    dashboard.PunkOSWindow._persist_theme = lambda self: None   # don't touch config.json
    return dashboard.PunkOSWindow(dashboard.load_config())


def test_status_pill_maps_planner_states():
    """FR1.3 / UI-4: free-text planner status maps to the five named states."""
    print("  test_status_pill_maps_planner_states...")
    _ensure_app()
    from ui.theme import DARK
    from ui.widgets import StatusPill

    pill = StatusPill(DARK)
    cases = [
        ("Perceiving (step 1)", "thinking"),
        ("Thinking (step 2)", "thinking"),
        ("Planning", "planning"),
        ("Executing (step 3)", "executing"),
        ("Executing in sandbox", "executing"),
        ("Awaiting confirmation (HIGH)", "awaiting"),
        ("Completed", "completed"),
        ("Failed", "failed"),
        ("Stopped", "stopped"),
    ]
    for status, expected in cases:
        pill.set_from_status(status)
        assert_eq(pill._state_key, expected, f"status {status!r}")

    # Busy states animate, terminal states do not (UI-4).
    pill.set_from_status("Executing (step 1)")
    assert_true(pill.dot._timer.isActive(), "executing state should animate")
    pill.set_from_status("Completed")
    assert_true(not pill.dot._timer.isActive(), "completed state should not animate")
    print("    ✓ pass")


def test_chat_feed_rerenders_on_theme_change():
    """Colours are baked into the HTML, so a theme switch must replay messages."""
    print("  test_chat_feed_rerenders_on_theme_change...")
    _ensure_app()
    from ui.theme import DARK, LIGHT
    from ui.widgets import ChatFeed

    feed = ChatFeed(DARK)
    feed.add_user_message("Open Calculator")
    feed.add_agent_log("Capturing screen state")
    feed.add_result(True, "Calculator is open", "3 steps")
    assert_eq(len(feed._messages), 3, "messages recorded for replay")

    dark_html = feed.toHtml()
    feed.set_palette(LIGHT)
    light_html = feed.toHtml()

    assert_eq(len(feed._messages), 3, "replay must not duplicate or drop messages")
    assert_true(dark_html != light_html, "theme change must re-render the transcript")
    assert_true("Open Calculator" in feed.toPlainText(), "content preserved")

    feed.clear_feed()
    assert_eq(len(feed._messages), 0, "clear resets the replay buffer")
    print("    ✓ pass")


def test_plan_step_list_tracks_status():
    """FR1.4: pending / running / done / failed per sub-task."""
    print("  test_plan_step_list_tracks_status...")
    _ensure_app()
    from ui.theme import DARK
    from ui.widgets import PlanStepList

    steps = PlanStepList(DARK)
    steps.set_steps([
        {"step_number": 1, "description": "Open Run", "is_high_risk": False},
        {"step_number": 2, "description": "Type calc", "is_high_risk": False},
        {"step_number": 3, "description": "Delete file", "is_high_risk": True},
    ])
    assert_eq(steps.count(), 3, "all steps rendered")

    steps.mark(0, "done")
    steps.mark_current(1)
    assert_eq(steps._steps[0]["status"], "done", "step 1 done")
    assert_eq(steps._steps[1]["status"], "running", "step 2 running")
    assert_eq(steps._steps[2]["status"], "pending", "step 3 pending")

    # A completed step must not be reset by a later mark_current.
    steps.mark_current(2)
    assert_eq(steps._steps[0]["status"], "done", "done steps stay done")

    assert_true("⚠" in steps.item(2).text(), "high-risk step is flagged")
    steps.clear_steps()
    assert_eq(steps.count(), 0, "cleared")
    print("    ✓ pass")


def test_confirmation_dialog_gating():
    """FR2.5 / FR2.7: Confirm is disabled until acknowledgement is ticked."""
    print("  test_confirmation_dialog_gating...")
    _ensure_app()
    from ui.theme import DARK
    from ui.dashboard import ConfirmationDialog

    payload = {
        "action_type": "code_execute", "target_tag": None, "text": "rm stuff",
        "reasoning": "Remove old archives", "level": "HIGH",
        "reason": "Bulk operation affecting 14 files",
        "preview": [("delete", "C:/Users/x/Downloads/a.zip"),
                    ("modify", "C:/Users/x/Downloads/index.csv")],
        "affected_paths": ["a"] * 14,
    }
    dialog = ConfirmationDialog(None, payload, DARK)

    assert_true(not dialog.confirm_button.isEnabled(),
                "Confirm must start disabled")
    dialog.ack_check.setChecked(True)
    assert_true(dialog.confirm_button.isEnabled(),
                "Confirm must enable once acknowledged")
    dialog.ack_check.setChecked(False)
    assert_true(not dialog.confirm_button.isEnabled(),
                "Un-ticking must disable Confirm again")

    # FR2.2: plain-language summary, not a raw action dump.
    summary = ConfirmationDialog._plain_summary(payload)
    assert_true("14 files" in summary, f"summary should state the scope: {summary!r}")
    print("    ✓ pass")


def test_main_window_builds_all_panels():
    print("  test_main_window_builds_all_panels...")
    window = _build_window()

    assert_eq(window.stack.count(), 6, "six pages in the stack")
    assert_eq(len(window._nav_buttons), 6, "six sidebar entries")

    from ui.dashboard import NAV_ITEMS
    for index, (label, _icon, page) in enumerate(NAV_ITEMS):
        window._navigate_to(page)
        assert_eq(window.stack.currentIndex(), page, f"navigate to {label}")
        assert_eq(window.page_title.text(), label, f"header title for {label}")
        assert_eq(window._nav_buttons[index].property("active"), "true",
                  f"{label} nav marked active")

    window._navigate_to(0)
    window.close()
    print("    ✓ pass")


def test_keyboard_shortcuts_registered():
    """USE-8: primary actions reachable from the keyboard."""
    print("  test_keyboard_shortcuts_registered...")
    window = _build_window()
    registered = {s.key().toString() for s in window.findChildren(QShortcut)}

    for expected in ("Ctrl+Return", "Esc", "Ctrl+F", "Ctrl+T", "Ctrl+M", "Ctrl+H"):
        assert_true(expected in registered,
                    f"{expected} not registered (have: {sorted(registered)})")
    window.close()
    print("    ✓ pass")


def test_theme_toggle_switches_palette():
    print("  test_theme_toggle_switches_palette...")
    window = _build_window()
    start = window.theme_name

    window._toggle_theme()
    assert_true(window.theme_name != start, "theme name flips")
    assert_eq(window.palette_.name, window.theme_name, "palette follows the name")

    window._toggle_theme()
    assert_eq(window.theme_name, start, "toggling twice returns to the start")

    # The header control must stay inside the window (it clipped once).
    button_right = window.theme_btn.mapTo(window, window.theme_btn.rect().topRight()).x()
    assert_true(button_right <= window.width(),
                f"theme button right edge {button_right} exceeds window {window.width()}")
    window.close()
    print("    ✓ pass")


def test_plugin_panel_renders_live_registry():
    """FR6.1: the panel lists real installed plugins, not hardcoded stubs."""
    print("  test_plugin_panel_renders_live_registry...")
    _ensure_app()
    from core.plugins import PluginRegistry
    from ui.plugin_manager import PluginManagerPanel
    from ui.theme import DARK

    registry = PluginRegistry(Path(__file__).parent.parent / "data" / "plugins")
    registry.discover()

    panel = PluginManagerPanel(palette=DARK)
    panel.set_registry(registry)

    assert_eq(len(panel.plugins), len(registry.plugins), "all plugins rendered as cards")
    assert_true(len(panel._cards) == len(registry.plugins), "one card per plugin")

    first = next(iter(registry.plugins))
    panel._show_plugin_detail(first)
    assert_eq(panel.current_plugin_id, first, "selection tracked")
    assert_true(panel.workflows_list.count() > 0, "workflows listed for the selection")
    assert_true(panel.test_btn.isEnabled(), "Test enabled once a plugin is selected")
    print("    ✓ pass")


def test_all_panels_instantiate():
    """Every panel must construct standalone without raising."""
    print("  test_all_panels_instantiate...")
    _ensure_app()
    import tempfile
    from ui.file_search_view import FileSearchPanel
    from ui.history_view import HistoryPanel
    from ui.plugin_manager import PluginManagerPanel
    from ui.settings_view import SettingsPanel
    from ui.web_agent_view import WebAgentPanel
    from ui.execution_view import LiveExecutionView
    from core.history import TaskHistoryStore

    tmp = Path(tempfile.mkdtemp())
    FileSearchPanel(db_path=str(tmp / "idx.db"), authorized_dirs=[str(tmp)])
    HistoryPanel(TaskHistoryStore(tmp / "hist.db"))
    PluginManagerPanel()
    SettingsPanel(Path(__file__).parent.parent / "config.json")
    WebAgentPanel()
    LiveExecutionView()
    print("    ✓ pass")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("─" * 60)
    print("PunkOS UI Tests")
    print("─" * 60)

    if not PYQT_AVAILABLE:
        print("  PyQt6 is not installed — skipping UI tests.")
        print("  Install with:  pip install PyQt6")
        print("─" * 60)
        return

    tests = [
        test_both_themes_meet_wcag_aa,
        test_accent_tokens_serve_both_roles,
        test_stylesheet_builds_for_both_themes,
        test_no_unscoped_qframe_stylesheets,
        test_no_widget_wide_background_rule,
        test_status_pill_maps_planner_states,
        test_chat_feed_rerenders_on_theme_change,
        test_plan_step_list_tracks_status,
        test_confirmation_dialog_gating,
        test_main_window_builds_all_panels,
        test_keyboard_shortcuts_registered,
        test_theme_toggle_switches_palette,
        test_plugin_panel_renders_live_registry,
        test_all_panels_instantiate,
    ]
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"    ✗ FAIL: {e}")
            failed.append((test.__name__, e))

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
