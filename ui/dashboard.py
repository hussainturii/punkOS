"""
PunkOS main window.

Layout follows SDS UI-1: one unified window with a persistent sidebar, a main
content area, and a status bar that is visible from every panel.

  sidebar          Dashboard / File Search / History / Plugins / Web / Settings
  header           agent status pill (UI-4, FR1.3) + theme toggle
  dashboard page   conversation feed + command bar | plan & live tabs
  status bar       provider, mode, keyboard hints

The planner runs on a worker QThread and talks to the UI through Qt signals, so
widgets are only ever touched from the GUI thread.
"""
from __future__ import annotations

import json
import sys
import traceback
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QListWidget, QListWidgetItem,
    QFrame, QSplitter, QMessageBox, QDialog, QDialogButtonBox, QCheckBox,
    QStatusBar, QSizePolicy, QStackedWidget, QTabWidget, QScrollArea,
)

from core.perception import VisualPerceptionEngine
from core.reasoning import ReasoningCore, NextAction, TaskPlan
from core.controller import DesktopController
from core.safety import SafetySystem, RiskAssessment, RiskLevel
from core.history import TaskHistoryStore
from core.planner import StrategicPlanner, TaskResult
from core.llm import build_provider
from core.bootstrap import build_subsystems
from core.window_handoff import DesktopHandoff
from core.task_policy import result_presentation

from ui.theme import METRICS, MONO_STACK, PALETTES, DARK, build_stylesheet
from ui.widgets import (
    ChatFeed, EmptyState, MicButton, PlanStepList, RiskBadge, StatusPill, Toast,
    caption, card, divider, heading, muted, restyle, section_label, set_variant,
)


# ============================================================================
# Worker thread
# ============================================================================

class AgentWorker(QObject):
    """Runs a single task in a background thread."""

    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    plan_signal = pyqtSignal(dict)
    step_signal = pyqtSignal(int, str, bool)
    perceive_signal = pyqtSignal(object, object)
    confirm_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(dict)

    confirm_response_received = pyqtSignal(bool)

    def __init__(self, planner: StrategicPlanner, goal: str):
        super().__init__()
        self.planner = planner
        self.goal = goal
        self._confirm_response: Optional[bool] = None
        self._confirmation_event = threading.Event()
        # The worker event loop is occupied by run(); a queued reply cannot run.
        # This direct slot only updates thread-safe confirmation state, never UI.
        self.confirm_response_received.connect(self._on_confirm_response, Qt.ConnectionType.DirectConnection)

    @pyqtSlot(bool)
    def _on_confirm_response(self, approved: bool):
        self._confirm_response = approved
        self._confirmation_event.set()

    def _wait_for_confirmation(self, action: NextAction, assessment: RiskAssessment) -> bool:
        self._confirm_response = None
        self._confirmation_event.clear()
        if self.planner.is_stopped():
            return False
        self.confirm_signal.emit({
            "action_type": action.action_type,
            "target_tag": action.target_tag,
            "text": action.text,
            "reasoning": action.reasoning,
            "level": assessment.level.value,
            "reason": assessment.reason,
            # FR2.3: the diff rows the safety layer computed.
            "preview": list(assessment.preview),
            "affected_paths": list(assessment.affected_paths),
        })
        # Wait up to 5 minutes. FR2.5: the modal must never auto-close on
        # timeout into an approval — falling through returns False (cancel).
        for _ in range(300 * 10):
            if self.planner.is_stopped():
                return False
            if self._confirmation_event.wait(0.1):
                return bool(self._confirm_response) and not self.planner.is_stopped()
        return False

    @pyqtSlot()
    def run(self):
        try:
            self.planner.on_log = self.log_signal.emit
            self.planner.on_status = self.status_signal.emit
            self.planner.on_confirm = self._wait_for_confirmation
            self.planner.on_plan_ready = lambda p: self.plan_signal.emit(p.to_dict())
            self.planner.on_step_complete = lambda i, a, r: self.step_signal.emit(
                i, f"{a.action_type} -> {r.message}", r.success
            )
            self.planner.on_perceive = lambda state, target: self.perceive_signal.emit(state, target)

            result = self.planner.run(self.goal)
            self.finished_signal.emit({
                "task_id": result.task_id,
                "success": result.success,
                "final_message": result.final_message,
                "steps_taken": result.steps_taken,
                "duration_seconds": result.duration_seconds,
                "execution_mode": result.execution_mode,
                "learning": result.learning,
            })
        except Exception as e:
            self.log_signal.emit(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
            self.finished_signal.emit({
                "task_id": "error", "success": False, "final_message": str(e),
                "steps_taken": 0, "duration_seconds": 0.0,
                "execution_mode": "DESKTOP", "learning": None,
            })


# ============================================================================
# High-risk confirmation modal — UI-3, FR2.1-FR2.7
# ============================================================================

class ConfirmationDialog(QDialog):
    """Blocks execution until the user explicitly approves.

    FR2.2 plain-language summary, FR2.3 diff preview, FR2.4 risk badge,
    FR2.7 acknowledgement checkbox gating the Confirm button.
    """

    def __init__(self, parent, payload: dict, palette=DARK):
        super().__init__(parent)
        self.palette_ = palette
        level = payload.get("level", "HIGH")

        self.setWindowTitle(f"{level} risk action — confirmation required")
        self.setModal(True)
        self.setMinimumWidth(560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(*(METRICS.space_xl,) * 4)
        outer.setSpacing(METRICS.space_lg)

        # -- header: badge + headline --
        header = QHBoxLayout()
        header.setSpacing(METRICS.space_md)
        header.addWidget(RiskBadge(level, palette))
        title = QLabel("PunkOS paused before doing this")
        title.setProperty("role", "h3")
        header.addWidget(title)
        header.addStretch()
        outer.addLayout(header)

        # -- FR2.2: plain-language description --
        outer.addWidget(QLabel(self._plain_summary(payload)))
        summary = QLabel(payload.get("reason", ""))
        summary.setWordWrap(True)
        summary.setProperty("role", "muted")
        outer.addWidget(summary)

        outer.addWidget(divider())

        # -- FR2.3: diff preview --
        preview_rows = payload.get("preview") or []
        if preview_rows:
            outer.addWidget(section_label("What will change"))
            preview = QListWidget()
            preview.setMaximumHeight(150)
            colours = {
                "add": palette.success,
                "delete": palette.danger,
                "modify": palette.warning,
            }
            markers = {"add": "+", "delete": "−", "modify": "~"}
            for kind, detail in preview_rows:
                item = QListWidgetItem(f" {markers.get(kind, '?')}  {detail}")
                from PyQt6.QtGui import QColor
                item.setForeground(QColor(colours.get(kind, palette.text)))
                font = item.font()
                font.setFamily("Consolas")
                item.setFont(font)
                preview.addItem(item)
            outer.addWidget(preview)

        # -- technical detail, secondary --
        detail = QTextEdit()
        detail.setReadOnly(True)
        detail.setMaximumHeight(110)
        detail.setStyleSheet(
            f"background-color:{palette.surface_alt}; color:{palette.text_muted};"
            f"border:1px solid {palette.border}; border-radius:{METRICS.radius_md}px;"
            f"font-family:{MONO_STACK}; font-size:{METRICS.font_sm}pt;"
        )
        detail.setPlainText(
            f"action     {payload.get('action_type')}\n"
            f"target     tag [{payload.get('target_tag')}]\n"
            f"text       {payload.get('text', '')!r}\n"
            f"reasoning  {payload.get('reasoning', '')}"
        )
        outer.addWidget(detail)

        # -- FR2.7: acknowledgement gate --
        self.ack_check = QCheckBox("I understand this cannot be undone")
        outer.addWidget(self.ack_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        cancel_btn = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_btn.setText("Cancel")
        cancel_btn.setDefault(True)      # safest option is the default

        self.confirm_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.confirm_button.setText("Confirm && execute")
        set_variant(self.confirm_button, "danger")
        self.confirm_button.setEnabled(False)

        self.ack_check.toggled.connect(self.confirm_button.setEnabled)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    @staticmethod
    def _plain_summary(payload: dict) -> str:
        """USE-3: describe the action without jargon."""
        action = payload.get("action_type", "")
        text = payload.get("text", "")
        paths = payload.get("affected_paths") or []

        if paths:
            if len(paths) == 1:
                return f"This will modify 1 file: {paths[0]}"
            return f"This will affect {len(paths)} files."
        phrases = {
            "click": "The agent wants to click something that may be irreversible.",
            "type": f"The agent wants to type {text!r}.",
            "key": f"The agent wants to press {text!r}.",
            "code_execute": "The agent wants to run a Python script it wrote.",
            "drag": "The agent wants to drag one item onto another.",
        }
        return phrases.get(action, f"The agent wants to perform: {action}")


# ============================================================================
# Main window
# ============================================================================

NAV_ITEMS = [
    ("Dashboard",   "◈", 0),
    ("File Search", "◎", 1),
    ("History",     "◔", 2),
    ("Plugins",     "◇", 3),
    ("Web Agent",   "◍", 4),
    ("Settings",    "◌", 5),
]


class PunkOSWindow(QMainWindow):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.setWindowTitle("PunkOS — Agentic Desktop AI")
        self.resize(1360, 840)
        self.setMinimumSize(1040, 660)

        self.theme_name = config.get("ui", {}).get("theme", "dark")
        self.palette_ = PALETTES.get(self.theme_name, DARK)

        self.worker: Optional[AgentWorker] = None
        self.thread: Optional[QThread] = None
        self._nav_buttons: list[QPushButton] = []
        self._pending_log: list[str] = []

        self._apply_theme()
        self._build_ui()
        self._install_shortcuts()
        self._build_planner()

    # ---------------------------- theming ----------------------------

    def _apply_theme(self):
        self.palette_ = PALETTES.get(self.theme_name, DARK)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_stylesheet(self.palette_))

    def _toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self._apply_theme()

        # Widgets that paint their own surfaces need the new palette handed
        # over — the global stylesheet cannot reach colours set in Python.
        for widget in (self.status_pill, self.chat_view, self.plan_list,
                       self.mic_btn, self.toast, self.plugin_panel,
                       self.file_search_panel, self.history_panel,
                       self.web_panel, self.settings_panel):
            if widget is not None and hasattr(widget, "set_palette"):
                try:
                    widget.set_palette(self.palette_)
                except Exception:
                    pass

        self.theme_btn.setText("◐" if self.theme_name == "dark" else "◑")
        self.theme_btn.setToolTip(
            f"Switch to {'light' if self.theme_name == 'dark' else 'dark'} theme  (Ctrl+T)"
        )
        self._persist_theme()
        self.toast.show_message(f"{self.theme_name.title()} theme", "info", 1500)

    def _persist_theme(self):
        """Remember the choice, but never let a config write break the UI."""
        try:
            path = Path(__file__).parent.parent / "config.json"
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("ui", {})["theme"] = self.theme_name
            path.write_text(json.dumps(data, indent=4), encoding="utf-8")
        except Exception:
            pass

    # ---------------------------- layout ----------------------------

    def _build_ui(self):
        from ui.file_search_view import FileSearchPanel
        from ui.history_view import HistoryPanel
        from ui.settings_view import SettingsPanel
        from ui.plugin_manager import PluginManagerPanel
        from ui.web_agent_view import WebAgentPanel

        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_sidebar())

        # Right side: header + stacked pages
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._build_header())

        self.stack = QStackedWidget()
        self.stack.setContentsMargins(0, 0, 0, 0)

        # 0 — Dashboard
        self.stack.addWidget(self._build_dashboard_page())

        # 1 — File Search
        safety_cfg = self.config.get("safety", {})
        history_cfg = self.config.get("history", {})
        index_db = str(Path(history_cfg.get("db_path", "data/history/punkos.db")).parent
                       / "file_index.db")
        self.file_search_panel = FileSearchPanel(
            db_path=index_db,
            authorized_dirs=safety_cfg.get("authorized_directories", []),
        )
        self.stack.addWidget(self.file_search_panel)

        # 2 — History (lazy: needs history_store from _build_planner)
        self.history_panel: Optional[HistoryPanel] = None
        self._history_placeholder = QWidget()
        placeholder_layout = QVBoxLayout(self._history_placeholder)
        placeholder_layout.addWidget(EmptyState(
            "◔", "Task history",
            "Run a task from the Dashboard and it will appear here.", self.palette_))
        self.stack.addWidget(self._history_placeholder)

        # 3 — Plugins
        self.plugin_panel = PluginManagerPanel(palette=self.palette_)
        self.stack.addWidget(self.plugin_panel)

        # 4 — Web Agent
        self.web_panel = WebAgentPanel()
        self.stack.addWidget(self.web_panel)

        # 5 — Settings
        config_path = Path(__file__).parent.parent / "config.json"
        if not config_path.exists():
            config_path = Path("config.json")
        self.settings_panel = SettingsPanel(config_path)
        self.settings_panel.settings_saved.connect(self._on_settings_saved)
        self.stack.addWidget(self.settings_panel)

        right_layout.addWidget(self.stack, 1)
        outer.addWidget(right, 1)

        # Status bar — USE-2, always visible
        self.status_bar = QStatusBar()
        self.provider_label = QLabel("connecting…")
        self.provider_label.setProperty("role", "caption")
        self.mode_label = QLabel("")
        self.mode_label.setProperty("role", "caption")
        hint = QLabel("Ctrl+Enter run   ·   Esc stop   ·   Ctrl+F search   ·   Ctrl+T theme")
        hint.setProperty("role", "caption")
        self.status_bar.addWidget(hint)
        self.status_bar.addPermanentWidget(self.mode_label)
        self.status_bar.addPermanentWidget(self.provider_label)
        self.setStatusBar(self.status_bar)

        # Floating toast layer
        self.toast = Toast(self, self.palette_)

        self._set_active_nav(0)

    def _build_sidebar(self) -> QWidget:
        side = QFrame()
        side.setProperty("role", "sidebar")
        side.setFixedWidth(METRICS.sidebar_width)

        layout = QVBoxLayout(side)
        layout.setContentsMargins(METRICS.space_md, METRICS.space_xl,
                                  METRICS.space_md, METRICS.space_lg)
        layout.setSpacing(METRICS.space_xs)

        brand = QLabel("PunkOS")
        brand_font = QFont()
        brand_font.setPointSize(METRICS.font_xl)
        brand_font.setBold(True)
        brand.setFont(brand_font)
        brand.setStyleSheet(f"color:{self.palette_.accent}; padding: 0 8px;")
        layout.addWidget(brand)

        tagline = QLabel("Agentic Desktop AI")
        tagline.setProperty("role", "caption")
        tagline.setStyleSheet(f"color:{self.palette_.text_faint}; padding: 0 8px 16px 8px;")
        layout.addWidget(tagline)

        for label, icon, index in NAV_ITEMS:
            btn = QPushButton(f"  {icon}   {label}")
            set_variant(btn, "nav")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _c, i=index: self._navigate_to(i))
            layout.addWidget(btn)
            self._nav_buttons.append(btn)

        layout.addStretch()
        layout.addWidget(divider())

        self.sidebar_note = QLabel("Local model · offline")
        self.sidebar_note.setProperty("role", "caption")
        self.sidebar_note.setWordWrap(True)
        self.sidebar_note.setStyleSheet(f"color:{self.palette_.text_faint}; padding: 8px;")
        layout.addWidget(self.sidebar_note)
        return side

    def _build_header(self) -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(METRICS.space_xl, METRICS.space_lg,
                                  METRICS.space_xl, METRICS.space_md)
        layout.setSpacing(METRICS.space_md)

        self.page_title = QLabel("Dashboard")
        self.page_title.setProperty("role", "h2")
        layout.addWidget(self.page_title)
        layout.addStretch()

        self.status_pill = StatusPill(self.palette_)
        layout.addWidget(self.status_pill)

        # Geometric glyph rather than ☾/☀ — the moon and sun emoji fall back to
        # an emoji font at a different metric and get clipped in a fixed-size
        # button. These render in the UI font, like the sidebar icons.
        self.theme_btn = QPushButton("◐")
        set_variant(self.theme_btn, "ghost")
        self.theme_btn.setFixedSize(34, 34)
        self.theme_btn.setToolTip("Switch to light theme  (Ctrl+T)")
        self.theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_btn.clicked.connect(self._toggle_theme)
        layout.addWidget(self.theme_btn)
        return header

    def _build_dashboard_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(METRICS.space_xl, 0, METRICS.space_xl, METRICS.space_lg)
        layout.setSpacing(METRICS.space_lg)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        conversation = self._build_conversation_panel()
        inspector = self._build_inspector_panel()
        conversation.setMinimumWidth(420)
        inspector.setMinimumWidth(320)
        splitter.addWidget(conversation)
        splitter.addWidget(inspector)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)
        # Stretch factors only govern *extra* space; without explicit sizes the
        # initial split comes from sizeHint, which handed the tab widget most of
        # the window and squeezed the conversation into a narrow column.
        splitter.setSizes([760, 500])
        layout.addWidget(splitter)
        return page

    def _build_conversation_panel(self) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(METRICS.space_md)

        self.chat_view = ChatFeed(self.palette_)
        layout.addWidget(self.chat_view, 1)
        self._show_welcome()

        # -- command bar (UI-2) --
        bar = QFrame()
        bar.setProperty("role", "card")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(METRICS.space_md, METRICS.space_md,
                                      METRICS.space_md, METRICS.space_md)
        bar_layout.setSpacing(METRICS.space_sm)

        self.mic_btn = MicButton(self.palette_)
        self.mic_btn.clicked.connect(self._on_mic_clicked)
        bar_layout.addWidget(self.mic_btn)

        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText(
            "Describe a task in plain English — e.g. “open Calculator and compute 25 × 4”"
        )
        self.command_input.setMaxLength(2000)          # FR1.1
        self.command_input.setMinimumHeight(METRICS.control_height + 6)
        self.command_input.returnPressed.connect(self._on_send_clicked)
        bar_layout.addWidget(self.command_input, 1)

        self.send_btn = QPushButton("Run")
        set_variant(self.send_btn, "primary")
        self.send_btn.setMinimumWidth(92)
        self.send_btn.setMinimumHeight(METRICS.control_height + 6)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.clicked.connect(self._on_send_clicked)
        bar_layout.addWidget(self.send_btn)

        self.stop_btn = QPushButton("Stop")
        set_variant(self.stop_btn, "danger")
        self.stop_btn.setMinimumWidth(80)
        self.stop_btn.setMinimumHeight(METRICS.control_height + 6)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        bar_layout.addWidget(self.stop_btn)

        layout.addWidget(bar)
        return wrap

    def _build_inspector_panel(self) -> QWidget:
        from ui.execution_view import LiveExecutionView

        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()

        # -- Plan tab --
        plan_tab = QWidget()
        plan_layout = QVBoxLayout(plan_tab)
        plan_layout.setContentsMargins(*(METRICS.space_lg,) * 4)
        plan_layout.setSpacing(METRICS.space_sm)

        self.plan_summary = QLabel("No active task.")
        self.plan_summary.setWordWrap(True)
        self.plan_summary.setProperty("role", "muted")
        plan_layout.addWidget(self.plan_summary)

        self.plan_list = PlanStepList(self.palette_)
        plan_layout.addWidget(self.plan_list, 1)
        tabs.addTab(plan_tab, "Plan")

        # -- Live tab --
        self.live_view = LiveExecutionView()
        tabs.addTab(self.live_view, "Live")

        layout.addWidget(tabs)
        self.right_tabs = tabs
        return wrap

    def _show_welcome(self):
        self.chat_view.add_system_note(
            "PunkOS is ready. Type a task below, or press Ctrl+M to speak one.", "info")
        self.chat_view.add_agent_log(
            "Examples:  “Open Notepad and type Hello World”  ·  "
            "“Organise my Downloads folder”  ·  “Go to example.com”")

    # ---------------------------- shortcuts (USE-8) ----------------------------

    def _install_shortcuts(self):
        def bind(sequence: str, handler):
            QShortcut(QKeySequence(sequence), self, activated=handler)

        bind("Ctrl+Return", self._on_send_clicked)
        bind("Ctrl+Enter", self._on_send_clicked)
        bind("Esc", self._on_stop_clicked)
        bind("Ctrl+F", lambda: self._navigate_to(1))
        bind("Ctrl+H", lambda: self._navigate_to(2))
        bind("Ctrl+T", self._toggle_theme)
        bind("Ctrl+M", self._on_mic_clicked)
        bind("Ctrl+L", self._clear_conversation)
        bind("Ctrl+,", lambda: self._navigate_to(5))

    def _clear_conversation(self):
        self.chat_view.clear_feed()
        self._show_welcome()

    # ---------------------------- planner setup ----------------------------

    def _build_planner(self):
        try:
            self.provider = build_provider(self.config)
            self.provider_label.setText(self.provider.name())
        except Exception as e:
            self.provider_label.setText("LLM unavailable")
            self.chat_view.add_system_note(f"LLM provider error: {e}", "danger")
            QMessageBox.critical(
                self, "LLM provider unavailable",
                f"{e}\n\nFix the issue and reopen Settings to retry.")
            self.send_btn.setEnabled(False)
            return

        self.send_btn.setEnabled(True)
        agent_cfg = self.config.get("agent", {})
        safety_cfg = self.config.get("safety", {})
        history_cfg = self.config.get("history", {})

        self.perception = VisualPerceptionEngine(
            max_elements=agent_cfg.get("max_elements", 40),
            max_dimension=agent_cfg.get("screenshot_max_dimension", 1280),
        )
        self.reasoning = ReasoningCore(self.provider)
        self.controller = DesktopController()

        self.subsystems = build_subsystems(
            self.config,
            project_root=Path(__file__).resolve().parent.parent,
            on_log=self.chat_view.add_agent_log,
        )
        for warning in self.subsystems.warnings:
            self.chat_view.add_system_note(warning, "warning")

        self.safety = SafetySystem(
            authorized_dirs=safety_cfg.get("authorized_directories", []),
            blocked_paths=safety_cfg.get("blocked_paths", []),
            confirm_medium_risk=safety_cfg.get("confirm_medium_risk", True),
            audit_log=self.subsystems.audit_log,
        )
        self.history_store = TaskHistoryStore(
            history_cfg.get("db_path", "data/history/punkos.db"))

        self._planner_args = {
            "planning_mode": agent_cfg.get('planning_mode', 'auto'),
            "max_task_seconds": agent_cfg.get('max_task_seconds', 600),
            "desktop_handoff": DesktopHandoff(),
            "perception": self.perception,
            "reasoning": self.reasoning,
            "controller": self.controller,
            "safety": self.safety,
            "history_store": self.history_store,
            "max_steps": agent_cfg.get("max_steps_per_task", 12),
            "max_consecutive_failures": agent_cfg.get("max_consecutive_failures", 3),
            "step_delay": agent_cfg.get("step_delay_seconds", 1.5),
            "use_planning_phase": agent_cfg.get("use_planning_phase", True),
            "use_memory_recall": agent_cfg.get('use_memory_recall', False),
            **self.subsystems.as_planner_kwargs(),
        }

        if self.subsystems.plugins is not None:
            self.plugin_panel.set_registry(self.subsystems.plugins)
            count = len(self.subsystems.plugins.plugins)
            self.sidebar_note.setText(f"{self.provider.name()}\n{count} plugins loaded")
        if self.subsystems.web_agent is not None:
            self.web_panel.set_agent(self.subsystems.web_agent)

    # ---------------------------- navigation ----------------------------

    def _set_active_nav(self, page_idx: int):
        for i, btn in enumerate(self._nav_buttons):
            btn.setProperty("active", "true" if i == page_idx else "false")
            restyle(btn)
        if 0 <= page_idx < len(NAV_ITEMS):
            self.page_title.setText(NAV_ITEMS[page_idx][0])

    def _navigate_to(self, page_idx: int):
        # History needs history_store, which only exists after _build_planner.
        if page_idx == 2 and self.history_panel is None and hasattr(self, "history_store"):
            from ui.history_view import HistoryPanel
            self.history_panel = HistoryPanel(
                self.history_store, memory=getattr(self.subsystems, "memory", None))
            self.history_panel.replay_task.connect(self._on_replay_task)
            idx = self.stack.indexOf(self._history_placeholder)
            self.stack.removeWidget(self._history_placeholder)
            self.stack.insertWidget(idx, self.history_panel)
            self._history_placeholder.deleteLater()

        if page_idx == 2 and self.history_panel is not None:
            self.history_panel.refresh()

        self.stack.setCurrentIndex(page_idx)
        self._set_active_nav(page_idx)

    # ---------------------------- task lifecycle ----------------------------

    def _on_send_clicked(self):
        goal = self.command_input.text().strip()
        if not goal:                                    # FR1.1: non-empty
            self.toast.show_message("Type a task first", "warning", 1600)
            return
        if self.thread is not None and self.thread.isRunning():
            # FR1.6: no duplicate submission while executing.
            self.toast.show_message("A task is already running", "warning")
            return
        if not hasattr(self, "_planner_args"):
            self.toast.show_message("LLM provider is not configured", "danger")
            return

        self.command_input.clear()
        self.chat_view.add_user_message(goal)
        self.plan_list.clear_steps()
        self.plan_summary.setText("Planning…")
        self.live_view.reset()
        self.right_tabs.setCurrentIndex(1)

        planner = StrategicPlanner(**self._planner_args)
        self.worker = AgentWorker(planner, goal)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.worker.log_signal.connect(self.chat_view.add_agent_log)
        self.worker.status_signal.connect(self._on_status_update)
        self.worker.plan_signal.connect(self._on_plan_ready)
        self.worker.step_signal.connect(self._on_step_complete)
        self.worker.step_signal.connect(self.live_view.append_action)
        self.worker.perceive_signal.connect(self.live_view.update_perception)
        self.worker.confirm_signal.connect(self._on_confirm_request)
        self.worker.finished_signal.connect(self._on_task_finished)

        self.thread.started.connect(self.worker.run)
        self._active_planner = planner

        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.thread.start()

    def _on_stop_clicked(self):
        planner = getattr(self, "_active_planner", None)
        if planner is None or self.thread is None or not self.thread.isRunning():
            return
        planner.request_stop()
        self.chat_view.add_system_note("Stop requested — cancelling the active request and preventing further actions.", "warning")
        self.toast.show_message("Stopping…", "warning", 1800)

    @pyqtSlot(str)
    def _on_status_update(self, status: str):
        self.status_pill.set_from_status(status)

    @pyqtSlot(dict)
    def _on_plan_ready(self, plan_dict: dict):
        self.plan_summary.setText(plan_dict.get("summary", ""))
        self.plan_list.set_steps(plan_dict.get("steps", []))
        self.plan_list.mark_current(0)
        self._plan_step_index = 0

    @pyqtSlot(int, str, bool)
    def _on_step_complete(self, step_num: int, description: str, success: bool):
        # An input receipt cannot mark a high-level plan step complete.
        # The live action log already displays execution success/failure.
        pass

    @pyqtSlot(dict)
    def _on_confirm_request(self, payload: dict):
        worker = self.worker
        if worker is None or worker.planner.is_stopped():
            return
        dialog = ConfirmationDialog(self, payload, self.palette_)
        self._confirmation_dialog = dialog
        approved = dialog.exec() == QDialog.DialogCode.Accepted
        self._confirmation_dialog = None
        self.chat_view.add_system_note(
            f"{payload['level']} risk action {'approved' if approved else 'cancelled'} by user.",
            "warning" if approved else "info")
        worker.confirm_response_received.emit(approved)

    @pyqtSlot(dict)
    def _on_task_finished(self, result: dict):
        dialog = getattr(self, '_confirmation_dialog', None)
        if dialog is not None:
            dialog.reject()
        meta = (f"{result['steps_taken']} steps · {result['duration_seconds']:.1f}s · "
                f"{result.get('execution_mode', 'DESKTOP')} mode")
        self.chat_view.add_result(result["success"], result["final_message"], meta)

        if result.get("learning"):
            self.chat_view.add_system_note(result["learning"], "info")

        state, title, severity = result_presentation(result["success"], result["final_message"])
        self.status_pill.set_state(state)
        self.toast.show_message(title, severity)

        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(5000)
            self.thread = None
            self.worker = None

        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ---------------------------- voice (USE-6) ----------------------------

    def _on_mic_clicked(self):
        if not hasattr(self, "_voice") or self._voice is None:
            try:
                from core.voice import VoiceInput
                self._voice = VoiceInput()
            except Exception as e:
                QMessageBox.warning(
                    self, "Voice input unavailable",
                    f"{e}\n\nInstall with:  pip install faster-whisper sounddevice")
                return

        if self._voice.is_recording:
            self.mic_btn.set_recording(False)
            self.toast.show_message("Transcribing…", "info", 1500)
            try:
                audio = self._voice.stop_recording()
                from core.voice import TranscribeThread
                self._transcribe = TranscribeThread(
                    self._voice, audio, self._on_transcription_ready)
                self._transcribe.start()
            except Exception as e:
                self.chat_view.add_system_note(f"Transcription failed: {e}", "danger")
        else:
            try:
                self._voice.start_recording()
                self.mic_btn.set_recording(True)
                self.toast.show_message("Recording — click again to stop", "info", 2000)
            except Exception as e:
                QMessageBox.warning(self, "Microphone unavailable", str(e))

    def _on_transcription_ready(self, text: str):
        # FR1.2 business rule: transcription lands in the field for review,
        # it is never auto-dispatched.
        if text:
            self.command_input.setText(text)
            self.command_input.setFocus()
        else:
            self.chat_view.add_system_note("Nothing was transcribed.", "warning")

    # ---------------------------- misc ----------------------------

    @pyqtSlot()
    def _on_settings_saved(self):
        try:
            self.config = load_config()
        except Exception as e:
            QMessageBox.warning(self, "Config reload failed", str(e))
            return
        try:
            self._build_planner()
            self.toast.show_message("Settings saved", "success")
        except Exception as e:
            QMessageBox.warning(
                self, "Provider rebuild failed",
                f"Settings saved, but the LLM provider could not be rebuilt:\n\n{e}")

    @pyqtSlot(str)
    def _on_replay_task(self, goal: str):
        """FR7.2: a replay is a fresh execution, not a cached result."""
        self._navigate_to(0)
        self.command_input.setText(goal)
        self._on_send_clicked()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "toast") and self.toast.isVisible():
            self.toast._reposition()

    def closeEvent(self, event):
        """Tear down background work so nothing is orphaned (SEC-7, REL-6)."""
        planner = getattr(self, "_active_planner", None)
        if planner is not None:
            planner.request_stop()

        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            if not self.thread.wait(3000):
                event.ignore()
                self.toast.show_message('Task is stopping. Close again after it finishes.', 'warning')
                return

        if not self.plugin_panel.shutdown():
            event.ignore()
            self.toast.show_message('Knowledge indexing is finishing. Close again after it finishes.', 'warning')
            return
        if not self.web_panel.shutdown():
            event.ignore()
            self.toast.show_message('Browser is stopping. Close again after it finishes.', 'warning')
            return

        super().closeEvent(event)


# ============================================================================
# Entry point
# ============================================================================

def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.json"
    if not config_path.exists():
        config_path = Path("config.json")
    return json.loads(config_path.read_text(encoding="utf-8"))


def main():
    # main.py normally creates the QApplication first, so that Qt claims the
    # process DPI awareness before `uiautomation` does (see main.py's docstring).
    # Reuse that instance when it exists; construct one when this module is run
    # directly, e.g. `python ui/dashboard.py`.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        app.setApplicationName("PunkOS")

    config = load_config()
    window = PunkOSWindow(config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
