"""
UI-5: Web Agent Control Panel (SDS 4.5.5, FR5.1-FR5.7).

Layout mirrors the SDS mockup M-5:

  left   — URL bar, live page preview (FR5.1), tab strip (FR5.2)
  right  — tabbed: DOM Explorer (FR5.3) / Form Fill (FR5.4, FR5.6) /
           Scrape (FR5.7) / Script (FR5.5)

Every Playwright call runs on a worker QThread. Playwright's sync API blocks,
and blocking the GUI thread would freeze the window for the length of a page
load — so the panel stays responsive by construction.
"""
from __future__ import annotations

import base64
from queue import Queue
from typing import Optional

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QTextEdit, QPlainTextEdit, QFrame, QSplitter, QMessageBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
)

from ui.theme import METRICS, MONO_STACK, DARK
from ui.widgets import muted, section_label, set_variant


class WebWorker(QThread):
    """Own all synchronous Playwright calls on one persistent thread."""

    completed = pyqtSignal(object, object)

    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.commands = Queue()

    def submit(self, command):
        self.commands.put(command)

    def run(self):
        # Keep the Python/greenlet stack alive between commands. Returning from
        # a Qt slot between sync Playwright calls can invalidate its greenlet.
        while True:
            command = self.commands.get()
            self.execute(command)
            if command[0] == 'shutdown':
                break

    def execute(self, command):
        from core.web_agent import WebResult
        method, args, kwargs = command
        snapshot = {}
        try:
            if method in ('navigate', 'new_tab', 'switch_tab', 'close', 'shutdown'):
                self.agent.form_fields = []
            if method not in ('close', 'shutdown') and not self.agent.is_running:
                result = self.agent.start()
                if not result.success:
                    self.completed.emit(result, snapshot)
                    return
            if method in ('close', 'shutdown'):
                result = self.agent.close()
            elif method == 'dom_snapshot':
                root = self.agent.dom_snapshot()
                result = WebResult(root is not None, 'DOM refreshed' if root else 'DOM unavailable', root)
            elif method == 'discover_form_fields':
                fields = self.agent.discover_form_fields()
                error = getattr(self.agent, 'last_discovery_error', '')
                result = WebResult(not bool(error), error or f'Found {len(fields)} fillable field(s) on {self.agent.current_page_label()}', fields)
            else:
                result = getattr(self.agent, method)(*args, **kwargs)
            if method != 'shutdown':
                snapshot = {'script': self.agent.generated_script(),
                            'fields': self.agent.form_fields,
                            'running': self.agent.is_running}
                if snapshot['running']:
                    snapshot.update(preview=self.agent.screenshot_b64(),
                                    tabs=self.agent.list_tabs(),
                                    gate=self.agent.detect_human_gate() if method == 'navigate' else None)
        except Exception as exc:
            result = WebResult(False, f'Web agent error: {exc}')
        self.completed.emit(result, snapshot)


class WebAgentPanel(QWidget):
    """The full Web Agent view."""

    def __init__(self, agent=None):
        super().__init__()
        self.agent = agent
        self._thread: Optional[QThread] = None
        self._worker: Optional[WebWorker] = None
        self._busy = False
        self._then = None
        self._snapshot = {}
        self._closing = False
        self._scraped: list[dict] = []
        self._setup_ui()

    def set_agent(self, agent) -> None:
        self.agent = agent
        self._set_status("Browser ready — enter a URL to begin.", DARK.success)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(METRICS.space_xl, 0, METRICS.space_xl, METRICS.space_lg)
        main.setSpacing(METRICS.space_sm)

        # No title label here — the window header already names the page.
        main.addWidget(muted(
            "Drives a real browser through the page DOM, which is more reliable "
            "than reading the screen."
        ))

        # ---- URL bar ----
        url_row = QHBoxLayout()
        url_row.setSpacing(METRICS.space_sm)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("example.com   or   https://example.com")
        self.url_input.setMinimumHeight(METRICS.control_height)
        self.url_input.returnPressed.connect(self._on_navigate)
        url_row.addWidget(self.url_input, 1)

        self.go_btn = QPushButton("Go")
        set_variant(self.go_btn, "primary")
        self.go_btn.setMinimumWidth(70)
        self.go_btn.clicked.connect(self._on_navigate)
        url_row.addWidget(self.go_btn)

        self.newtab_btn = QPushButton("New tab")
        self.newtab_btn.clicked.connect(self._on_new_tab)
        url_row.addWidget(self.newtab_btn)

        self.close_btn = QPushButton("Close browser")
        set_variant(self.close_btn, "danger")
        self.close_btn.clicked.connect(self._on_close_browser)
        url_row.addWidget(self.close_btn)
        main.addLayout(url_row)

        self.status_label = QLabel("Browser not started.")
        self.status_label.setProperty("role", "caption")
        main.addWidget(self.status_label)

        # ---- split: preview | inspector ----
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: preview + tabs
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(section_label("Live preview"))
        self.preview = QLabel("No page loaded.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(320)
        self.preview.setStyleSheet(
            f"background:{DARK.surface}; border:1px solid {DARK.border};"
            f"border-radius:{METRICS.radius_lg}px; color:{DARK.text_faint};"
        )
        self.preview.setScaledContents(False)
        left_layout.addWidget(self.preview, 1)

        left_layout.addWidget(section_label("Open tabs  ·  5 maximum"))
        self.tab_list = QListWidget()
        self.tab_list.setMaximumHeight(110)
        self.tab_list.itemClicked.connect(self._on_tab_clicked)
        left_layout.addWidget(self.tab_list)

        splitter.addWidget(left)

        # Right: inspector tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_dom_tab(), "DOM")
        self.tabs.addTab(self._build_form_tab(), "Form Fill")
        self.tabs.addTab(self._build_scrape_tab(), "Scrape")
        self.tabs.addTab(self._build_script_tab(), "Script")
        splitter.addWidget(self.tabs)

        splitter.setSizes([560, 460])
        main.addWidget(splitter, 1)

    def _build_dom_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(muted("The live element tree of the page Playwright is driving."))

        refresh = QPushButton("Refresh DOM")
        refresh.clicked.connect(self._on_refresh_dom)
        refresh_row = QHBoxLayout()
        refresh_row.addWidget(refresh)
        refresh_row.addStretch()
        layout.addLayout(refresh_row)

        self.dom_tree = QTreeWidget()
        self.dom_tree.setHeaderLabels(["Element", "Selector"])
        self.dom_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.dom_tree, 1)
        return w

    def _build_form_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(muted(
            "Fields are matched on their label, placeholder or name, so missing "
            "or generated element ids are fine. One 'key: value' pair per line."
        ))

        self.form_values = QPlainTextEdit()
        self.form_values.setPlaceholderText("email: ada@example.com\nfull name: Ada Lovelace")
        self.form_values.setMaximumHeight(120)
        layout.addWidget(self.form_values)

        row = QHBoxLayout()
        discover = QPushButton("Discover fields")
        discover.clicked.connect(self._on_discover_fields)
        row.addWidget(discover)

        fill = QPushButton("Fill form")
        set_variant(fill, "primary")
        fill.clicked.connect(self._on_fill_form)
        row.addWidget(fill)

        verify = QPushButton("Verify")
        verify.clicked.connect(self._on_verify_form)
        row.addWidget(verify)
        layout.addLayout(row)

        layout.addWidget(section_label("Fill status"))
        self.field_table = QTableWidget(0, 3)
        self.field_table.setHorizontalHeaderLabels(["Field", "Status", "Detail"])
        self.field_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.field_table, 1)
        return w

    def _build_scrape_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(muted(
            "Pull structured records out of the page. "
            "One 'column: css-selector' pair per line."
        ))

        self.scrape_schema = QPlainTextEdit()
        self.scrape_schema.setPlaceholderText("title: h2.product-title\nprice: span.price")
        self.scrape_schema.setMaximumHeight(110)
        layout.addWidget(self.scrape_schema)

        row = QHBoxLayout()
        run = QPushButton("Scrape")
        set_variant(run, "primary")
        run.clicked.connect(self._on_scrape)
        row.addWidget(run)

        self.export_json_btn = QPushButton("Export JSON")
        self.export_json_btn.clicked.connect(lambda: self._export("json"))
        self.export_json_btn.setEnabled(False)
        row.addWidget(self.export_json_btn)

        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(lambda: self._export("csv"))
        self.export_csv_btn.setEnabled(False)
        row.addWidget(self.export_csv_btn)
        layout.addLayout(row)

        self.scrape_table = QTableWidget(0, 0)
        layout.addWidget(self.scrape_table, 1)
        return w

    def _build_script_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(muted(
            "The Playwright script equivalent to everything run so far, "
            "so you can verify exactly what was executed."
        ))
        self.script_view = QTextEdit()
        self.script_view.setReadOnly(True)
        self.script_view.setFontFamily("Consolas")
        self.script_view.setStyleSheet(
            f"background:{DARK.surface}; color:{DARK.text}; font-family:{MONO_STACK};"
            f"border:1px solid {DARK.border}; border-radius:{METRICS.radius_md}px;"
        )
        layout.addWidget(self.script_view, 1)
        return w

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _dispatch(self, method: str, *args, then=None, **kwargs):
        """Run a WebAgent method on a worker thread."""
        if self.agent is None:
            QMessageBox.warning(self, "Web agent unavailable",
                                "The web agent is not configured. Check that Playwright "
                                "is installed and 'web.enabled' is true in config.json.")
            return
        if self._closing or self._busy:
            self._set_status("Busy — waiting for the previous action to finish.", DARK.warning)
            return

        self._set_status(f"Running {method}…", DARK.accent)
        if method in ('navigate', 'new_tab', 'switch_tab', 'close'):
            # Inspections belong to the previous page, even if the requested
            # transition later fails. Require a fresh inspection before use.
            self.dom_tree.clear()
            self.field_table.setRowCount(0)
            self._scraped = []
            self.scrape_table.setRowCount(0)
            self.scrape_table.setColumnCount(0)
            self.export_json_btn.setEnabled(False)
            self.export_csv_btn.setEnabled(False)
            self._snapshot = {}
        if self._thread is None:
            self._worker = WebWorker(self.agent)
            self._thread = self._worker
            self._worker.completed.connect(self._on_worker_done)
            self._thread.start()
        self._busy = True
        self._then = then
        self._worker.submit((method, args, kwargs))

    @pyqtSlot(object, object)
    def _on_worker_done(self, result, snapshot):
        self._busy = False
        then, self._then = self._then, None
        self._snapshot = snapshot
        if self._closing:
            return
        self._set_status(result.message, DARK.success if result.success else DARK.danger)
        self._refresh_preview()
        self._refresh_tabs()
        self._refresh_script()
        if callable(then):
            then(result)

    def shutdown(self, timeout_ms=3000):
        """Queue cleanup on the owner; never terminate a running browser call."""
        if self._thread is None or not self._thread.isRunning():
            return True
        if not self._closing:
            self._closing = True
            self._worker.submit(('shutdown', (), {}))
        return self._thread.wait(timeout_ms)

    def _on_navigate(self):
        url = self.url_input.text().strip()
        if not url:
            return
        self._dispatch("navigate", url, then=self._after_navigate)

    def _after_navigate(self, result):
        if not result.success:
            return
        gate = self._snapshot.get('gate')
        if gate:
            QMessageBox.information(
                self, "Manual step needed",
                f"{gate} detected on this page.\n\n"
                "PunkOS does not attempt to solve CAPTCHAs or authentication "
                "challenges by design. Complete it in the browser window, then continue."
            )
        self._on_refresh_dom()

    def _on_new_tab(self):
        self._dispatch("new_tab")

    def _on_close_browser(self):
        self._dispatch('close')

    def _on_tab_clicked(self, item: QListWidgetItem):
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is not None:
            self._dispatch("switch_tab", int(index))

    def _on_refresh_dom(self):
        self._dispatch('dom_snapshot', then=self._render_dom)

    def _render_dom(self, result):
        self.dom_tree.clear()
        if result.success and result.data is not None:
            self.dom_tree.addTopLevelItem(self._dom_item(result.data))
            self.dom_tree.expandToDepth(1)

    def _dom_item(self, node) -> QTreeWidgetItem:
        label = f"<{node.tag}>"
        if node.node_id:
            label += f" #{node.node_id}"
        if node.text:
            label += f"  “{node.text[:40]}”"
        item = QTreeWidgetItem([label, node.selector])
        for child in node.children:
            item.addChild(self._dom_item(child))
        return item

    def _on_discover_fields(self):
        self._dispatch('discover_form_fields',
                       then=lambda r: self._render_fields(r.data or []))

    def _on_fill_form(self):
        values = self._parse_pairs(self.form_values.toPlainText())
        if not values:
            QMessageBox.information(self, "Nothing to fill",
                                    "Add at least one 'key: value' line first.")
            return
        self._dispatch("fill_form", values,
                       then=lambda r: self._render_fields(self._snapshot.get('fields', [])))

    def _on_verify_form(self):
        self._dispatch('verify_fill')

    def _render_fields(self, fields):
        self.field_table.setRowCount(len(fields))
        colors = {
            "Completed": "#7fdb7f", "Failed": "#e57373",
            "In Progress": "#ffd54f", "Pending": "#888",
        }
        for row, fld in enumerate(fields):
            name = fld.label or fld.name or fld.selector
            status_item = QTableWidgetItem(fld.status.value)
            from PyQt6.QtGui import QColor
            status_item.setForeground(QColor(colors.get(fld.status.value, "#ddd")))
            detail = fld.error or fld.value or ("required" if fld.required else "")
            self.field_table.setItem(row, 0, QTableWidgetItem(name))
            self.field_table.setItem(row, 1, status_item)
            self.field_table.setItem(row, 2, QTableWidgetItem(detail))

    def _on_scrape(self):
        schema = self._parse_pairs(self.scrape_schema.toPlainText())
        if not schema:
            QMessageBox.information(self, "No schema",
                                    "Add at least one 'column: selector' line first.")
            return
        self._dispatch("scrape", schema, then=self._render_scrape)

    def _render_scrape(self, result):
        self._scraped = result.data if (result.success and result.data) else []
        self.export_json_btn.setEnabled(bool(self._scraped))
        self.export_csv_btn.setEnabled(bool(self._scraped))
        if not self._scraped:
            self.scrape_table.setRowCount(0)
            self.scrape_table.setColumnCount(0)
            return
        columns = list(self._scraped[0].keys())
        self.scrape_table.setColumnCount(len(columns))
        self.scrape_table.setHorizontalHeaderLabels(columns)
        self.scrape_table.setRowCount(len(self._scraped))
        for r, record in enumerate(self._scraped):
            for c, col in enumerate(columns):
                self.scrape_table.setItem(r, c, QTableWidgetItem(str(record.get(col, ""))))

    def _export(self, fmt: str):
        if not self._scraped:
            return
        default = f"scrape.{fmt}"
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {fmt.upper()}", default, f"{fmt.upper()} files (*.{fmt})"
        )
        if not path:
            return
        from core.web_agent import WebAgent
        content = (WebAgent.to_csv(self._scraped) if fmt == "csv"
                   else WebAgent.to_json(self._scraped))
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(content)
            self._set_status(f"Exported {len(self._scraped)} record(s) to {path}", DARK.success)
        except OSError as e:
            QMessageBox.warning(self, "Export failed", str(e))

    # ------------------------------------------------------------------
    # Refreshers
    # ------------------------------------------------------------------

    def _refresh_preview(self):
        self.preview.setPixmap(QPixmap())
        self.preview.setText('No page preview available.')
        b64 = self._snapshot.get('preview', '')
        if not b64:
            return
        pixmap = QPixmap()
        pixmap.loadFromData(base64.b64decode(b64))
        if pixmap.isNull():
            return
        self.preview.setPixmap(pixmap.scaled(
            self.preview.width(), self.preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _refresh_tabs(self):
        self.tab_list.clear()
        for tab in self._snapshot.get('tabs', []):
            marker = "▶ " if tab.status == "active" else "   "
            item = QListWidgetItem(f"{marker}{tab.index + 1}. {tab.title[:44]}\n      {tab.url[:60]}")
            item.setData(Qt.ItemDataRole.UserRole, tab.index)
            self.tab_list.addItem(item)

    def _refresh_script(self):
        if self.agent is None:
            return
        self.script_view.setPlainText(self._snapshot.get('script', ''))

    def _set_status(self, text: str, color: str = ""):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color or DARK.text_muted};")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pairs(text: str) -> dict[str, str]:
        """Parse 'key: value' lines, ignoring blanks and comments."""
        out: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            if key and value:
                out[key] = value
        return out
