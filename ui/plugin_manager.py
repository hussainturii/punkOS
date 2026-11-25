"""
UI-7: Plugin Manager (SDS 4.5.7, FR6.1-FR6.6).

Left: installed plugins as cards. Right: configuration for the selected plugin —
its registered workflows, knowledge base, and the Test action.

Styling note: every card scopes its stylesheet with an objectName selector
(`QFrame#pluginCard`). A bare `QFrame { border: ... }` set on a parent leaks
into every child QLabel, because QLabel is itself a QFrame subclass — which is
what previously made version numbers and headings render as bordered input
boxes.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QListWidget, QListWidgetItem, QScrollArea, QMessageBox, QSplitter,
    QFileDialog, QSizePolicy,
)

from ui.theme import METRICS, DARK
from ui.widgets import (
    EmptyState, caption, divider, heading, muted, section_label, set_variant,
)


# Fallback content when the panel is opened without a live registry (e.g. when
# running this view standalone). The real list comes from core.plugins.
STUB_PLUGINS = [
    {
        "id": "example",
        "name": "No registry attached",
        "version": "0.0.0",
        "icon": "◇",
        "description": "Plugins load from data/plugins once the app is fully started.",
        "active": False,
        "workflows": [],
    },
]


class PluginCard(QFrame):
    """One plugin in the installed list."""

    selected = pyqtSignal(str)
    toggle_active = pyqtSignal(str, bool)

    def __init__(self, plugin: dict, palette=DARK):
        super().__init__()
        self.plugin = plugin
        self.palette_ = palette
        self.setObjectName("pluginCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._apply_style(selected=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*(METRICS.space_md,) * 4)
        layout.setSpacing(METRICS.space_sm)

        # -- title row --
        top = QHBoxLayout()
        top.setSpacing(METRICS.space_md)

        icon = QLabel(plugin["icon"])
        icon.setStyleSheet(f"font-size: {METRICS.font_xl}pt; border: none;")
        icon.setFixedWidth(30)
        top.addWidget(icon)

        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        name = QLabel(plugin["name"])
        name_font = QFont()
        name_font.setBold(True)
        name.setFont(name_font)
        name.setStyleSheet(f"color:{palette.text}; border: none;")
        title_col.addWidget(name)

        version = QLabel(f"v{plugin['version']}")
        version.setStyleSheet(
            f"color:{palette.text_faint}; font-size:{METRICS.font_xs}pt; border: none;")
        title_col.addWidget(version)
        top.addLayout(title_col, 1)

        self.status_chip = QLabel()
        self.status_chip.setFixedHeight(20)
        top.addWidget(self.status_chip, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(top)

        # -- description --
        desc = QLabel(plugin["description"])
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color:{palette.text_muted}; font-size:{METRICS.font_sm}pt; border: none;")
        layout.addWidget(desc)

        # -- actions --
        actions = QHBoxLayout()
        actions.setSpacing(METRICS.space_sm)

        configure = QPushButton("Configure")
        set_variant(configure, "primary")
        configure.setCursor(Qt.CursorShape.PointingHandCursor)
        configure.clicked.connect(lambda: self.selected.emit(plugin["id"]))
        actions.addWidget(configure)

        self.toggle_btn = QPushButton()
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.clicked.connect(self._on_toggle)
        actions.addWidget(self.toggle_btn)
        actions.addStretch()
        layout.addLayout(actions)

        self._render_state()

    # ------------------------------------------------------------------

    def _apply_style(self, selected: bool) -> None:
        border = self.palette_.accent if selected else self.palette_.border
        self.setStyleSheet(
            f"QFrame#pluginCard {{"
            f" background-color:{self.palette_.surface};"
            f" border:1px solid {border};"
            f" border-radius:{METRICS.radius_lg}px; }}"
            f"QFrame#pluginCard:hover {{ border-color:{self.palette_.accent}; }}"
        )

    def set_selected(self, selected: bool) -> None:
        self._apply_style(selected)

    def _render_state(self) -> None:
        active = self.plugin["active"]
        colour = self.palette_.success if active else self.palette_.text_faint
        self.status_chip.setText("  Active  " if active else "  Inactive  ")
        self.status_chip.setStyleSheet(
            f"color:{self.palette_.text_on_accent}; background-color:{colour};"
            f"border-radius:{METRICS.radius_pill}px; padding:2px 8px;"
            f"font-size:{METRICS.font_xs}pt; font-weight:700;"
        )
        self.toggle_btn.setText("Disable" if active else "Enable")

    def _on_toggle(self) -> None:
        self.plugin["active"] = not self.plugin["active"]
        self._render_state()
        self.toggle_active.emit(self.plugin["id"], self.plugin["active"])


class KnowledgeIndexWorker(QThread):
    result_ready = pyqtSignal(bool, str)

    def __init__(self, registry, plugin_id, parent=None):
        super().__init__(parent)
        self.registry, self.plugin_id = registry, plugin_id

    def run(self):
        try:
            ok, message = self.registry.index_knowledge_base(self.plugin_id, self.registry.knowledge_engine)
        except Exception as exc:
            ok, message = False, f'Knowledge indexing failed: {exc}'
        self.result_ready.emit(ok, message)


class PluginManagerPanel(QWidget):
    """The full Plugin Manager view."""

    ICONS = {"vscode": "◧", "excel": "▦", "notepad": "▤"}

    def __init__(self, registry=None, palette=DARK):
        super().__init__()
        self.registry = registry
        self._index_worker = None
        self.palette_ = palette
        self.current_plugin_id: Optional[str] = None
        self._cards: dict[str, PluginCard] = {}
        self.plugins = self._load_plugins()
        self._setup_ui()
        self._populate_plugin_list()

    # ---------------------- data ----------------------

    def _load_plugins(self) -> list:
        if self.registry is None:
            return list(STUB_PLUGINS)
        return [self._to_card_dict(p) for p in self.registry.plugins.values()]

    def _to_card_dict(self, plugin) -> dict:
        return {
            "id": plugin.plugin_id,
            "name": plugin.app_name,
            "version": plugin.version,
            "icon": self.ICONS.get(plugin.plugin_id, "◇"),
            "description": plugin.description,
            "active": plugin.is_active,
            "workflows": [
                {
                    "name": w.name.replace("_", " "),
                    "description": w.description,
                    "steps": len(w.steps),
                    "success_rate": w.success_rate,
                    "runs": w.execution_count,
                }
                for w in plugin.workflows
            ],
        }

    def set_registry(self, registry) -> None:
        """FR6.1: attach the live registry and re-render."""
        self.registry = registry
        self.reload()

    def set_palette(self, palette) -> None:
        """Follow a theme change — cards paint their own surfaces."""
        self.palette_ = palette
        self._populate_plugin_list()
        if self.current_plugin_id:
            self._show_plugin_detail(self.current_plugin_id)

    def reload(self) -> None:
        """FR6.1 business rule: reload after any install or uninstall."""
        self.plugins = self._load_plugins()
        self._populate_plugin_list()
        if self.current_plugin_id:
            self._show_plugin_detail(self.current_plugin_id)

    # ---------------------- layout ----------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(METRICS.space_xl, 0, METRICS.space_xl, METRICS.space_lg)
        root.setSpacing(METRICS.space_sm)

        root.addWidget(heading("Plugin Manager", 3))
        root.addWidget(muted(
            "Application-specific expertise for individual programs. A plugin stores "
            "the exact steps for a workflow so the agent doesn't have to rediscover "
            "them. Plugins are declarative data, never executable code."
        ))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # -- left: installed list --
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, METRICS.space_md, 0, 0)
        left_layout.setSpacing(METRICS.space_sm)
        left_layout.addWidget(section_label("Installed"))

        self.list_scroll = QScrollArea()
        self.list_scroll.setWidgetResizable(True)
        self.list_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.list_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, METRICS.space_sm, 0)
        self.list_layout.setSpacing(METRICS.space_md)
        self.list_layout.addStretch()
        self.list_scroll.setWidget(self.list_container)
        left_layout.addWidget(self.list_scroll, 1)

        store_btn = QPushButton("Browse plugin store")
        store_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        store_btn.clicked.connect(self._on_browse_store)
        left_layout.addWidget(store_btn)
        splitter.addWidget(left)

        # -- right: configuration --
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(METRICS.space_lg, METRICS.space_md, 0, 0)
        right_layout.setSpacing(METRICS.space_sm)

        self.detail_title = QLabel("No plugin selected")
        self.detail_title.setProperty("role", "h3")
        right_layout.addWidget(self.detail_title)

        self.detail_description = QLabel(
            "Choose a plugin on the left to see its workflows.")
        self.detail_description.setWordWrap(True)
        self.detail_description.setProperty("role", "muted")
        right_layout.addWidget(self.detail_description)

        right_layout.addWidget(divider())
        right_layout.addWidget(section_label("Registered workflows"))

        self.workflows_list = QListWidget()
        self.workflows_list.setWordWrap(True)
        right_layout.addWidget(self.workflows_list, 1)

        right_layout.addWidget(section_label("Knowledge base"))
        self.kb_hint = QLabel("Add reference documents (PDF, TXT, MD, DOCX, JSON, CSV).")
        self.kb_hint.setProperty("role", "caption")
        right_layout.addWidget(self.kb_hint)

        button_row = QHBoxLayout()
        button_row.setSpacing(METRICS.space_sm)

        self.kb_btn = QPushButton("Add document")
        self.kb_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.kb_btn.clicked.connect(self._on_add_knowledge)
        button_row.addWidget(self.kb_btn)
        self.index_btn = QPushButton('Build knowledge index')
        self.index_btn.clicked.connect(self._on_index_knowledge)
        button_row.addWidget(self.index_btn)

        self.test_btn = QPushButton("Validate plugin")
        set_variant(self.test_btn, "primary")
        self.test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.test_btn.clicked.connect(self._on_test_plugin)
        button_row.addWidget(self.test_btn)
        button_row.addStretch()
        right_layout.addLayout(button_row)

        splitter.addWidget(right)
        splitter.setSizes([420, 560])
        root.addWidget(splitter, 1)

        self._set_detail_enabled(False)

    def _set_detail_enabled(self, enabled: bool) -> None:
        self.index_btn.setEnabled(enabled and not (self._index_worker and self._index_worker.isRunning()))
        self.kb_btn.setEnabled(enabled)
        self.test_btn.setEnabled(enabled)

    # ---------------------- rendering ----------------------

    def _populate_plugin_list(self):
        # Drop the old cards but keep the trailing stretch.
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()

        if not self.plugins:
            empty = EmptyState("◇", "No plugins installed",
                               "Plugin manifests live in data/plugins.", self.palette_)
            self.list_layout.insertWidget(0, empty)
            return

        for index, plugin in enumerate(self.plugins):
            card = PluginCard(plugin, self.palette_)
            card.selected.connect(self._show_plugin_detail)
            card.toggle_active.connect(self._on_toggle_active)
            self.list_layout.insertWidget(index, card)
            self._cards[plugin["id"]] = card

    def _show_plugin_detail(self, plugin_id: str):
        plugin = next((p for p in self.plugins if p["id"] == plugin_id), None)
        if plugin is None:
            return
        self.current_plugin_id = plugin_id

        for pid, card in self._cards.items():
            card.set_selected(pid == plugin_id)

        self.detail_title.setText(f"{plugin['icon']}  {plugin['name']}  ·  v{plugin['version']}")
        self.detail_description.setText(plugin["description"])

        self.workflows_list.clear()
        workflows = plugin.get("workflows", [])
        if not workflows:
            self.workflows_list.addItem("This plugin registers no workflows.")
        for wf in workflows:
            if isinstance(wf, str):        # stub fallback shape
                self.workflows_list.addItem(wf)
                continue
            runs = wf["runs"]
            stat = (f"{wf['success_rate']:.0%} over {runs} run{'s' if runs != 1 else ''}"
                    if runs else "not run yet")
            item = QListWidgetItem(
                f"{wf['name']}  —  {wf['steps']} steps, {stat}\n    {wf['description']}")
            self.workflows_list.addItem(item)

        kb_count = 0
        if self.registry is not None:
            live = self.registry.plugins.get(plugin_id)
            kb_count = len(live.knowledge_base) if live else 0
        self.kb_hint.setText(
            f"{kb_count} document(s) attached. Supported: PDF, TXT, MD, DOCX, JSON, CSV."
        )
        self._set_detail_enabled(True)

    # ---------------------- handlers ----------------------

    def _on_toggle_active(self, plugin_id: str, active: bool):
        """Persist enable/disable to the plugin's manifest on disk."""
        if self.registry is None:
            return
        ok, message = self.registry.set_active(plugin_id, active)
        if not ok:
            self.reload()
            QMessageBox.warning(self, "Could not change plugin state", message)
            return
        for p in self.plugins:
            if p["id"] == plugin_id:
                p["active"] = active

    def _on_add_knowledge(self):
        """FR6.3: attach a reference document to the plugin's knowledge base."""
        if self.current_plugin_id is None or self.registry is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Add knowledge document", "",
            "Documents (*.pdf *.txt *.md *.docx *.json *.csv)")
        if not path:
            return
        ok, message = self.registry.add_knowledge_file(self.current_plugin_id, path)
        if ok:
            self.reload()
            QMessageBox.information(self, "Document added", message)
        else:
            QMessageBox.warning(self, "Could not add document", message)

    def _on_browse_store(self):
        QMessageBox.information(
            self, "Plugin store",
            "A hosted plugin store is not part of this build.\n\n"
            "Plugins are installed by dropping a folder containing a plugin.json "
            "manifest into data/plugins. Every manifest is schema-validated before "
            "it loads, and invalid plugins are rejected with the reason logged."
        )

    def _on_index_knowledge(self):
        if self._index_worker and self._index_worker.isRunning():
            return
        if self.registry is None or self.current_plugin_id is None or self.registry.knowledge_engine is None:
            QMessageBox.warning(self, 'Index unavailable', 'No knowledge search engine is attached.')
            return
        self.index_btn.setEnabled(False)
        self.kb_hint.setText('Indexing registered documents…')
        self._index_worker = KnowledgeIndexWorker(self.registry, self.current_plugin_id, self)
        self._index_worker.result_ready.connect(self._on_index_result)
        self._index_worker.finished.connect(lambda: self.index_btn.setEnabled(bool(self.current_plugin_id)))
        self._index_worker.start()

    def _on_index_result(self, ok, message):
        self.kb_hint.setText(message)
        if not ok:
            QMessageBox.warning(self, 'Indexing incomplete', message)

    def shutdown(self, timeout_ms=5000):
        return not self._index_worker or self._index_worker.wait(timeout_ms)

    def _on_test_plugin(self):
        """FR6.6: dry-run every workflow and report success rate."""
        if self.current_plugin_id is None:
            return
        plugin = next((p for p in self.plugins if p["id"] == self.current_plugin_id), None)
        if plugin is None:
            return
        if self.registry is None:
            QMessageBox.information(
                self, "Test plugin",
                "This panel has no plugin registry attached, so there is nothing to test.")
            return

        ok, report = self.registry.test_plugin(self.current_plugin_id)
        passed = sum(1 for line in report if line.startswith("PASS"))
        total = sum(1 for line in report if line.startswith(("PASS", "FAIL")))
        box = QMessageBox(self)
        box.setWindowTitle(f"Validation results — {plugin['name']}")
        box.setIcon(QMessageBox.Icon.Information if ok else QMessageBox.Icon.Warning)
        box.setText(f"{passed}/{total} workflow definitions valid")
        box.setInformativeText("Checks plugin definitions only. No application workflow was executed.")
        box.setDetailedText("\n".join(report))
        box.exec()
