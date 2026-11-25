"""
File Search panel — Week 3.

A standalone view that lets users:
- Index authorized directories (one-time setup, then incremental updates)
- Search files by content using natural language
- See relevance scores, content snippets, and extracted fields (Date / Amount / Vendor)

Indexing runs in a background thread to keep the UI responsive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QDesktopServices
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QFrame, QSizePolicy, QTextEdit,
    QProgressBar, QMessageBox, QComboBox,
)

from PyQt6.QtWidgets import QStackedWidget
from core.file_intel import FileSearchEngine, FileResult, IndexingError
from ui.theme import METRICS, DARK
from ui.widgets import EmptyState


# ============================================================================
# Worker threads
# ============================================================================

class IndexWorker(QObject):
    """Indexes directories in the background."""
    progress_signal = pyqtSignal(str)
    done_signal = pyqtSignal(int)        # files newly indexed
    error_signal = pyqtSignal(str)

    def __init__(self, engine: FileSearchEngine, directories: list[str]):
        super().__init__()
        self.engine = engine
        self.directories = directories

    @pyqtSlot()
    def run(self):
        total = 0
        # First, drop dead entries
        try:
            removed = self.engine.remove_missing()
        except Exception as e:
            self.error_signal.emit(f'Index cleanup failed: {e}')
            self.done_signal.emit(total)
            return
        if removed:
            self.progress_signal.emit(f"  cleaned up {removed} missing file(s)")

        for d in self.directories:
            if not d:
                continue
            self.progress_signal.emit(f"📁 Indexing {d}...")
            try:
                count = self.engine.index_directory(
                    d, on_progress=self.progress_signal.emit
                )
                total += count
                self.progress_signal.emit(f"   → {count} new file(s) added")
            except IndexingError as e:
                total += e.indexed
                self.error_signal.emit(f"Indexing incomplete for {d}: {e}")
            except Exception as e:
                self.error_signal.emit(f"Indexing failed for {d}: {e}")
        self.done_signal.emit(total)


class SearchWorker(QObject):
    """Runs a search query in the background."""
    done_signal = pyqtSignal(list)      # list[FileResult]
    error_signal = pyqtSignal(str)

    def __init__(self, engine: FileSearchEngine, query: str, path_filter: Optional[list[str]] = None,
                 status_filter: Optional[str] = None, search_mode: str = 'semantic'):
        super().__init__()
        self.engine = engine
        self.query = query
        self.path_filter = path_filter
        self.status_filter = status_filter
        self.search_mode = search_mode

    @pyqtSlot()
    def run(self):
        try:
            options = dict(path_filter=self.path_filter, status_filter=self.status_filter)
            if self.search_mode != 'semantic':
                options['search_mode'] = self.search_mode
            results = self.engine.search(self.query, **options)
        except Exception as e:
            self.error_signal.emit(f"Search failed: {e}")
            results = []
        self.done_signal.emit(results)


# ============================================================================
# Result item widget
# ============================================================================

class ResultItem(QFrame):
    """A single search result card."""
    def __init__(self, result: FileResult):
        super().__init__()
        self.result = result
        # Scoped by objectName: a bare `QFrame {...}` rule would also style every
        # child QLabel, since QLabel subclasses QFrame.
        self.setObjectName("resultCard")
        self.setStyleSheet(f"""
            QFrame#resultCard {{
                background-color: {DARK.surface};
                border: 1px solid {DARK.border};
                border-radius: {METRICS.radius_lg}px;
            }}
            QFrame#resultCard:hover {{ border-color: {DARK.accent}; }}
        """)

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(4)

        # Header: filename + score badge
        header = QHBoxLayout()
        path_obj = Path(result.path)
        name = QLabel(f"📄 {path_obj.name}")
        f = QFont(); f.setBold(True); f.setPointSize(10)
        name.setFont(f)
        name.setStyleSheet("color: #ddd;")
        header.addWidget(name, 1)

        mode = result.fields.get('search_mode', 'semantic')
        score = QLabel(f"Similarity {result.score:.2f}" if mode == 'semantic' else 'Literal match')
        score.setToolTip('Content similarity, not confidence that this document is correct or current.' if mode == 'semantic'
                         else 'Case-insensitive text match in the indexed filename or content.')
        score.setStyleSheet("""
            color: white;
            background-color: #4a7cd1;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 9pt;
            font-weight: bold;
        """)
        header.addWidget(score)
        v.addLayout(header)

        # Path
        path_lbl = QLabel(str(path_obj.parent))
        path_lbl.setStyleSheet("color: #888; font-family: Consolas, monospace; font-size: 8pt;")
        v.addWidget(path_lbl)

        # Snippet
        snippet = QLabel(result.snippet)
        snippet.setWordWrap(True)
        snippet.setStyleSheet("color: #bbb; padding: 4px 0; font-size: 9pt;")
        v.addWidget(snippet)

        # Fields chips
        chips_row = QHBoxLayout()
        chips_row.setSpacing(6)
        chips_row.addWidget(self._chip(f"Type: {result.file_type}", "#444"))
        if result.fields.get('declared_status'):
            chips_row.addWidget(self._chip(f"Declared status: {result.fields['declared_status']}", '#444'))
        if result.fields.get("date"):
            chips_row.addWidget(self._chip(f"📅 {result.fields['date']}", "#3a5e3a"))
        if result.fields.get("amount"):
            chips_row.addWidget(self._chip(f"💰 {result.fields['amount']}", "#5e3a3a"))
        if result.fields.get("vendor"):
            chips_row.addWidget(self._chip(f"🏢 {result.fields['vendor']}", "#3a3a5e"))
        chips_row.addStretch()
        v.addLayout(chips_row)

    @staticmethod
    def _chip(text: str, bg: str) -> QLabel:
        chip = QLabel(text)
        chip.setStyleSheet(f"""
            background-color: {bg};
            color: #ddd;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 8pt;
        """)
        return chip


# ============================================================================
# Main file search panel
# ============================================================================

class FileSearchPanel(QWidget):
    """Top-level widget for the File Search section of the dashboard."""

    def __init__(self, db_path: str, authorized_dirs: list[str]):
        super().__init__()
        self.authorized_dirs = authorized_dirs
        self.engine = FileSearchEngine(db_path)
        self.thread: Optional[QThread] = None
        self.worker: Optional[QObject] = None
        self._operation_error = ''
        self._setup_ui()
        self._refresh_status()

    # ---------------------- layout ----------------------

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(METRICS.space_xl, 0, METRICS.space_xl, METRICS.space_lg)
        main.setSpacing(METRICS.space_sm)

        # No title label — the window header already names the page.
        subtitle = QLabel(
            "Search indexed files by meaning, filename, or literal content. "
            "Build / update the index after adding or changing files."
        )
        subtitle.setProperty("role", "muted")
        subtitle.setWordWrap(True)
        main.addWidget(subtitle)

        # Index controls + status
        index_row = QHBoxLayout()
        self.index_btn = QPushButton("🔄  Build / update index")
        self.index_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 14px;
                background-color: #4a7cd1;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #5a8ce1; }
            QPushButton:disabled { background-color: #555; }
        """)
        self.index_btn.clicked.connect(self._on_index_clicked)
        index_row.addWidget(self.index_btn)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #aaa; padding-left: 10px;")
        index_row.addWidget(self.status_label, 1)
        main.addLayout(index_row)

        # Indexing log (hidden until indexing starts)
        self.index_log = QTextEdit()
        self.index_log.setReadOnly(True)
        self.index_log.setMaximumHeight(120)
        self.index_log.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a20;
                border: 1px solid #3a3a45;
                border-radius: 6px;
                font-family: Consolas, monospace;
                font-size: 9pt;
                padding: 6px;
                color: #ccc;
            }
        """)
        self.index_log.setVisible(False)
        main.addWidget(self.index_log)

        # Search row
        search_row = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText('Describe the file…  e.g. "invoice for $1500 from April"')
        self.query_input.setStyleSheet("""
            QLineEdit {
                padding: 12px;
                border: 1px solid #3a3a45;
                border-radius: 8px;
                background-color: #2a2a35;
                font-size: 11pt;
            }
        """)
        self.query_input.returnPressed.connect(self._on_search_clicked)
        search_row.addWidget(self.query_input, 1)

        self.search_btn = QPushButton("Search")
        self.search_btn.setMinimumWidth(100)
        self.search_btn.setStyleSheet("""
            QPushButton {
                padding: 12px 20px;
                background-color: #4a7cd1;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #5a8ce1; }
        """)
        self.search_btn.clicked.connect(self._on_search_clicked)
        search_row.addWidget(self.search_btn)
        main.addLayout(search_row)

        filter_row = QHBoxLayout()
        self.search_mode = QComboBox()
        self.search_mode.setAccessibleName('Search mode')
        for label, value in [('Meaning', 'semantic'), ('Filename contains', 'filename'), ('Content contains', 'content')]:
            self.search_mode.addItem(label, value)
        self.search_mode.setToolTip('Filename and content modes match literal text, ignoring case. Searches cover indexed files only; update the index after changing files.')
        self.search_mode.currentIndexChanged.connect(self._on_status_filter_changed)
        filter_row.addWidget(self.search_mode)
        filter_row.addWidget(QLabel('Declared document status:'))
        self.status_filter = QComboBox()
        self.status_filter.setAccessibleName('Declared document status')
        self.status_filter.addItem('Any status', None)
        for status in ('approved', 'confirmed', 'proposed', 'draft', 'rejected', 'superseded'):
            self.status_filter.addItem(status.capitalize(), status)
        self.status_filter.setToolTip('Matches an explicit Status: field in the indexed document. '
                                      'Does not verify approval or infer status from prose.')
        self.status_filter.currentIndexChanged.connect(self._on_status_filter_changed)
        filter_row.addWidget(self.status_filter)
        filter_row.addStretch()
        main.addLayout(filter_row)

        # Results
        self.results_list = QListWidget()
        self.results_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                outline: 0;
            }
            QListWidget::item { padding: 4px; }
            QListWidget::item:selected { background-color: transparent; }
        """)
        self.results_list.setSpacing(4)
        self.results_list.itemDoubleClicked.connect(self._on_result_double_clicked)

        # Results and the empty state occupy the same slot; only one shows at a
        # time, so an un-searched panel explains itself instead of being a void.
        self.results_stack = QStackedWidget()
        self.empty_state = EmptyState(
            "◎", "Nothing searched yet",
            "Build the index, then describe what you're looking for —\n"
            "for example “the invoice for $1,500 from April”.",
            DARK,
        )
        self.results_stack.addWidget(self.empty_state)
        self.results_stack.addWidget(self.results_list)
        main.addWidget(self.results_stack, 1)

        # Help footer
        help_lbl = QLabel("💡 Double-click a result to open the file's folder. Press Enter to search.")
        help_lbl.setStyleSheet("color: #777; font-size: 9pt; padding-top: 4px;")
        main.addWidget(help_lbl)

    # ---------------------- handlers ----------------------

    def _on_status_filter_changed(self):
        # Do not leave old results visible under a newly selected filter.
        self.results_list.clear()
        self.results_list.addItem(QListWidgetItem('Search filters changed. Press Search to update results.'))
        self.results_stack.setCurrentWidget(self.results_list)

    def _refresh_status(self):
        n = self.engine.file_count()
        self.status_label.setText(f"{n} file{'s' if n != 1 else ''} indexed")

    def _on_index_clicked(self):
        if self.thread is not None and self.thread.isRunning():
            return

        if not self.authorized_dirs:
            QMessageBox.warning(self, "No directories",
                                "No authorized directories configured. Edit config.json -> safety.authorized_directories.")
            return

        self.index_btn.setEnabled(False)
        self.search_btn.setEnabled(False)
        self._operation_error = ''
        self.index_log.setVisible(True)
        self.index_log.clear()
        self.index_log.append("Starting index build...")

        self.worker = IndexWorker(self.engine, self.authorized_dirs)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.progress_signal.connect(self.index_log.append)
        self.worker.error_signal.connect(self._on_operation_error)
        self.worker.done_signal.connect(self._on_index_done)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    @pyqtSlot(int)
    def _on_index_done(self, total: int):
        if self._operation_error:
            self.index_log.append(f"\nIndex incomplete. {total} new file(s) indexed; see errors above.")
        else:
            self.index_log.append(f"\n✅ Done. {total} new file(s) indexed.")
        try:
            self._refresh_status()
        except Exception as e:
            self.status_label.setText(f'Could not read index status: {e}')
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.worker = None
        self.index_btn.setEnabled(True)
        self.search_btn.setEnabled(True)

    @pyqtSlot(str)
    def _on_operation_error(self, message: str):
        self._operation_error = message
        if isinstance(self.worker, IndexWorker):
            self.index_log.append(message)

    def _on_search_clicked(self):
        if self.thread is not None and self.thread.isRunning():
            return
        query = self.query_input.text().strip()
        if not query:
            return
        try:
            count = self.engine.file_count()
        except Exception as e:
            QMessageBox.warning(self, 'Search unavailable', f'Could not read the index: {e}')
            return
        if count == 0:
            QMessageBox.information(self, "No index",
                                     "No files indexed yet. Click 'Build / update index' first.")
            return

        self.results_list.clear()
        self.results_list.addItem(QListWidgetItem("Searching..."))
        self.results_stack.setCurrentWidget(self.results_list)
        self._operation_error = ''
        self.search_btn.setEnabled(False)
        self.index_btn.setEnabled(False)

        # Search runs synchronously usually but for UX run async
        self.status_filter.setEnabled(False)
        self.search_mode.setEnabled(False)
        self.worker = SearchWorker(self.engine, query, path_filter=self.authorized_dirs,
                                   status_filter=self.status_filter.currentData(), search_mode=self.search_mode.currentData())
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.done_signal.connect(self._on_search_done)
        self.worker.error_signal.connect(self._on_operation_error)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    @pyqtSlot(list)
    def _on_search_done(self, results: list[FileResult]):
        self.status_filter.setEnabled(True)
        self.search_mode.setEnabled(True)
        self.results_list.clear()
        self.results_stack.setCurrentWidget(self.results_list)
        if self._operation_error:
            self.results_list.addItem(QListWidgetItem(self._operation_error))
        elif not results:
            empty = QListWidgetItem(
                "No matching files. Try different words or a different status filter, or update the index.")
            empty.setForeground(QColor(DARK.text_muted))
            self.results_list.addItem(empty)
        else:
            for r in results:
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, r.path)
                # Qt's item widget is painted over an otherwise empty list
                # item. UIA reads the item's accessibility role, not its labels.
                match_label = f'Similarity: {r.score:.2f}' if r.fields.get('search_mode','semantic') == 'semantic' else 'Literal text match'
                details = [Path(r.path).name, f'Path: {r.path}', match_label, f'Type: {r.file_type}']
                for key, label in (('declared_status', 'Declared status'),
                                   ('date', 'Date'), ('amount', 'Amount'), ('vendor', 'Vendor')):
                    if r.fields.get(key):
                        details.append(f'{label}: {r.fields[key]}')
                details.append(f'Content excerpt: {r.snippet}')
                item.setData(Qt.ItemDataRole.AccessibleTextRole, '\n'.join(details))
                item.setData(Qt.ItemDataRole.AccessibleDescriptionRole,
                             "Double-click to open this file's containing folder. "
                             "Similarity and declared status do not verify document correctness.")
                widget = ResultItem(r)
                item.setSizeHint(widget.sizeHint())
                self.results_list.addItem(item)
                self.results_list.setItemWidget(item, widget)

        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.worker = None

        self.search_btn.setEnabled(True)
        self.index_btn.setEnabled(True)

    def _on_result_double_clicked(self, item: QListWidgetItem):
        path_str = item.data(Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        path = Path(path_str)
        if not path.is_file():
            QMessageBox.warning(self, 'File unavailable', 'This file no longer exists or is unavailable. Update the index and search again.')
            return
        if not any(path.resolve().is_relative_to(Path(root).expanduser().resolve())
                   for root in self.authorized_dirs if str(root).strip()):
            QMessageBox.warning(self, 'File unavailable', 'This result is outside the current authorized folders. Search again with the current settings.')
            return
        # Open the file's containing folder
        folder = Path(path_str).parent
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            QMessageBox.warning(self, 'Could not open folder', 'Windows could not open the selected file’s containing folder.')
