"""
Task History panel — Week 4.

Displays the persistent SQLite task history with actions per row:
- ▶ Replay   — re-submit the same goal
- ★ Bookmark — mark as reusable workflow template
- 🗑 Delete  — remove the row from history

Also surfaces aggregate "Memory Statistics" (sessions stored, learned workflows,
memory usage) — fulfils the M-7 mockup from the SRS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QFrame, QMessageBox, QComboBox,
)

from core.history import TaskHistoryStore
from ui.theme import METRICS, DARK


class HistoryRowWidget(QFrame):
    """A single row card in the history list."""

    replay_requested = pyqtSignal(str)        # goal text
    bookmark_toggled = pyqtSignal(str, bool)  # task_id, is_bookmarked
    delete_requested = pyqtSignal(str)        # task_id

    def __init__(self, task: dict, is_bookmarked: bool = False):
        super().__init__()
        self.task_id = task["task_id"]
        self.goal_text = task["goal"]
        self.is_bookmarked = is_bookmarked

        success = bool(task["success"])
        # Scoped by objectName so the rule does not cascade into child QLabels
        # (QLabel is a QFrame subclass).
        self.setObjectName("historyRow")
        accent = DARK.success if success else DARK.danger
        self.setStyleSheet(f"""
            QFrame#historyRow {{
                background-color: {DARK.surface};
                border: 1px solid {DARK.border};
                border-left: 3px solid {accent};
                border-radius: {METRICS.radius_lg}px;
            }}
            QFrame#historyRow:hover {{ border-color: {DARK.border_strong};
                                       border-left-color: {accent}; }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Top: goal + status
        top = QHBoxLayout()
        status_icon = "✅" if success else "❌"
        status_lbl = QLabel(status_icon)
        status_lbl.setStyleSheet("font-size: 14pt;")
        top.addWidget(status_lbl)

        goal_lbl = QLabel(task["goal"])
        f = QFont(); f.setBold(True); f.setPointSize(10)
        goal_lbl.setFont(f)
        goal_lbl.setStyleSheet("color: #ddd;")
        goal_lbl.setWordWrap(True)
        top.addWidget(goal_lbl, 1)

        layout.addLayout(top)

        # Middle: metadata line
        meta_text = (
            f"⏱ {task['duration_seconds']:.1f}s   "
            f"🔢 {task['steps_taken']} steps   "
            f"⚡ {task['provider']}   "
            f"🕒 {task['created_at'][:16].replace('T', ' ')}"
        )
        meta_lbl = QLabel(meta_text)
        meta_lbl.setStyleSheet("color: #aaa; font-size: 8pt; font-family: Consolas, monospace;")
        layout.addWidget(meta_lbl)
        metrics = task.get('metrics') or {}
        categories = metrics.get('categories', {})
        if categories:
            model = categories.get('model', {})
            perception = categories.get('perception', {})
            timing = QLabel(f"Model: {model.get('calls', 0)} calls / {model.get('seconds', 0):.1f}s · "
                            f"Screen inspection: {perception.get('seconds', 0):.1f}s · "
                            f"Image calls: {model.get('image_calls', 0)}")
            timing.setWordWrap(True)
            timing.setToolTip('Counts are provider invocations; elapsed time includes internal retries. These are not token usage or billed costs.')
            layout.addWidget(timing)

        # Final message (if failed, show why)
        if task.get("final_message"):
            msg_lbl = QLabel(f"💬 {task['final_message']}")
            msg_lbl.setStyleSheet("color: #ccc; font-size: 9pt; padding-top: 2px;")
            msg_lbl.setWordWrap(True)
            layout.addWidget(msg_lbl)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        replay_btn = QPushButton("▶  Replay")
        replay_btn.setStyleSheet(self._btn_style("#4a7cd1"))
        replay_btn.clicked.connect(lambda: self.replay_requested.emit(self.goal_text))
        btn_row.addWidget(replay_btn)

        self.bookmark_btn = QPushButton("★ Bookmarked" if is_bookmarked else "☆ Bookmark")
        self.bookmark_btn.setStyleSheet(
            self._btn_style("#d1a04a" if is_bookmarked else "#444")
        )
        self.bookmark_btn.clicked.connect(self._toggle_bookmark)
        btn_row.addWidget(self.bookmark_btn)

        delete_btn = QPushButton("🗑  Delete")
        delete_btn.setStyleSheet(self._btn_style("#a04a4a"))
        delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.task_id))
        btn_row.addWidget(delete_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    @staticmethod
    def _btn_style(bg: str) -> str:
        return f"""
            QPushButton {{
                padding: 4px 10px;
                background-color: {bg};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 9pt;
            }}
            QPushButton:hover {{ background-color: {bg}; opacity: 0.8; }}
        """

    def _toggle_bookmark(self):
        self.is_bookmarked = not self.is_bookmarked
        self.bookmark_btn.setText("★ Bookmarked" if self.is_bookmarked else "☆ Bookmark")
        self.bookmark_btn.setStyleSheet(
            self._btn_style("#d1a04a" if self.is_bookmarked else "#444")
        )
        self.bookmark_toggled.emit(self.task_id, self.is_bookmarked)


class StatsTile(QFrame):
    """A single metric card in the stats row."""
    def __init__(self, icon: str, label: str, value: str):
        super().__init__()
        self.setObjectName("statsTile")
        self.setStyleSheet(f"""
            QFrame#statsTile {{
                background-color: {DARK.surface};
                border: 1px solid {DARK.border};
                border-radius: {METRICS.radius_lg}px;
            }}
        """)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(2)

        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("font-size: 18pt; color: #6fb1ff;")
        v.addWidget(icon_lbl)

        self.value_lbl = QLabel(value)
        f = QFont(); f.setBold(True); f.setPointSize(14)
        self.value_lbl.setFont(f)
        self.value_lbl.setStyleSheet("color: #ddd;")
        v.addWidget(self.value_lbl)

        label_lbl = QLabel(label)
        label_lbl.setStyleSheet("color: #888; font-size: 8pt;")
        v.addWidget(label_lbl)

    def update_value(self, value: str):
        self.value_lbl.setText(value)


class HistoryPanel(QWidget):
    """The full History view: stats, filters, scrollable list with row actions."""

    # Emitted when the user wants to re-run a past task; main window connects this
    # to its command pipeline.
    replay_task = pyqtSignal(str)

    def __init__(self, history_store: TaskHistoryStore, memory=None):
        super().__init__()
        self.store = history_store
        # FR7.1 business rule: history must persist across application restarts,
        # and so must the bookmarks on it. When procedural memory is available
        # they live in its bookmarks table; otherwise we degrade to in-memory.
        self.memory = memory
        self.bookmarks: set[str] = set()
        if self.memory is not None:
            try:
                self.bookmarks = self.memory.list_bookmarks()
            except Exception:
                self.bookmarks = set()
        self._current_filter = "All"
        self._setup_ui()
        self.refresh()

    # ---------------------- layout ----------------------

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(METRICS.space_xl, 0, METRICS.space_xl, METRICS.space_lg)
        main.setSpacing(METRICS.space_md)

        # No title label — the window header already names the page.
        sub = QLabel("Every task PunkOS has executed, across all sessions.")
        sub.setProperty("role", "muted")
        main.addWidget(sub)

        # Stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)
        self.stat_total = StatsTile("📊", "Total tasks", "0")
        self.stat_success = StatsTile("✅", "Successful", "0")
        self.stat_bookmarks = StatsTile("⭐", "Bookmarked", "0")
        self.stat_size = StatsTile("💾", "Memory used", "0 KB")
        for tile in (self.stat_total, self.stat_success, self.stat_bookmarks, self.stat_size):
            stats_row.addWidget(tile, 1)
        main.addLayout(stats_row)

        # Filter + refresh row
        filter_row = QHBoxLayout()
        filter_label = QLabel("Show:")
        filter_label.setStyleSheet("color: #aaa;")
        filter_row.addWidget(filter_label)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Successful only", "Failed only", "Bookmarked only"])
        self.filter_combo.setStyleSheet("""
            QComboBox {
                padding: 6px 10px;
                background-color: #2a2a35;
                border: 1px solid #3a3a45;
                border-radius: 4px;
                color: #ddd;
                min-width: 180px;
            }
        """)
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self.filter_combo)

        filter_row.addStretch()

        refresh_btn = QPushButton("🔄  Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 14px;
                background-color: #2a2a35;
                border: 1px solid #3a3a45;
                border-radius: 4px;
                color: #ddd;
            }
            QPushButton:hover { background-color: #3a3a45; }
        """)
        refresh_btn.clicked.connect(self.refresh)
        filter_row.addWidget(refresh_btn)
        main.addLayout(filter_row)

        # Task list
        self.task_list = QListWidget()
        self.task_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                outline: 0;
            }
            QListWidget::item:selected { background-color: transparent; }
        """)
        self.task_list.setSpacing(4)
        main.addWidget(self.task_list, 1)

    # ---------------------- handlers ----------------------

    def _on_filter_changed(self, text: str):
        self._current_filter = text
        self.refresh()

    def refresh(self):
        """Reload tasks from the DB and re-render the list."""
        self.task_list.clear()
        all_tasks = self.store.list_tasks(limit=200)

        filtered = self._apply_filter(all_tasks, self._current_filter)
        if not filtered:
            empty = QListWidgetItem("No tasks yet.  Run something from the Dashboard to get started.")
            empty.setForeground(QColor("#888"))
            empty.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.task_list.addItem(empty)
        else:
            for task in filtered:
                self._add_row(task)

        self._refresh_stats(all_tasks)

    def _apply_filter(self, tasks: list[dict], filter_label: str) -> list[dict]:
        if filter_label == "All":
            return tasks
        if filter_label == "Successful only":
            return [t for t in tasks if t["success"]]
        if filter_label == "Failed only":
            return [t for t in tasks if not t["success"]]
        if filter_label == "Bookmarked only":
            return [t for t in tasks if t["task_id"] in self.bookmarks]
        return tasks

    def _add_row(self, task: dict):
        widget = HistoryRowWidget(task, is_bookmarked=task["task_id"] in self.bookmarks)
        widget.replay_requested.connect(self._on_replay)
        widget.bookmark_toggled.connect(self._on_bookmark_toggled)
        widget.delete_requested.connect(self._on_delete)

        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        self.task_list.addItem(item)
        self.task_list.setItemWidget(item, widget)

    def _refresh_stats(self, all_tasks: list[dict]):
        self.stat_total.update_value(str(len(all_tasks)))
        success_count = sum(1 for t in all_tasks if t["success"])
        self.stat_success.update_value(str(success_count))
        self.stat_bookmarks.update_value(str(len(self.bookmarks)))

        # File size of the DB
        try:
            size_bytes = Path(self.store.db_path).stat().st_size
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            self.stat_size.update_value(size_str)
        except OSError:
            self.stat_size.update_value("—")

    # ---------------------- row callbacks ----------------------

    def _on_replay(self, goal: str):
        # Forward to dashboard via signal
        self.replay_task.emit(goal)

    def _on_bookmark_toggled(self, task_id: str, is_bookmarked: bool):
        if is_bookmarked:
            self.bookmarks.add(task_id)
        else:
            self.bookmarks.discard(task_id)
        # FR7.3: persist so the bookmark survives a restart.
        if self.memory is not None:
            try:
                self.memory.set_bookmark(task_id, is_bookmarked)
            except Exception as e:
                QMessageBox.warning(self, "Could not save bookmark", str(e))
        # Update stats without full re-render
        self.stat_bookmarks.update_value(str(len(self.bookmarks)))

    def _on_delete(self, task_id: str):
        confirm = QMessageBox.question(
            self,
            "Delete task?",
            "Remove this task from history? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.delete_task(task_id)
        except Exception as e:
            QMessageBox.warning(self, "Delete failed", str(e))
            return
        self.bookmarks.discard(task_id)
        if self.memory is not None:
            try:
                self.memory.set_bookmark(task_id, False)
            except Exception:
                pass
        self.refresh()
