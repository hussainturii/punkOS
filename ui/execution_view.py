"""
Live Execution View — Week 3.

A PyQt6 widget that shows what the agent currently sees and what it's targeting.
The widget exposes one slot, `update_perception(screen_state, target_tag)`, which
the dashboard wires up to the planner's `on_perceive` callback.
"""
from __future__ import annotations

import base64
import io
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QImage, QPixmap, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QListWidget,
    QListWidgetItem, QSizePolicy,
)

# We import lazily to avoid pulling in mss/uiautomation at module load on
# environments that don't have them yet.


class LiveExecutionView(QWidget):
    """Shows: live screen preview (with SoM tags + blue highlight on target),
    target element metadata, and a streaming action log."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._latest_pixmap: Optional[QPixmap] = None
        self._setup_ui()

    # ---------------------- layout ----------------------

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(8)

        # Header
        header = QLabel("👁  Live Execution View")
        f = QFont()
        f.setPointSize(11)
        f.setBold(True)
        header.setFont(f)
        main.addWidget(header)

        # Top row: screen preview (left) + metadata column (right)
        top_row = QHBoxLayout()

        # Screen preview frame
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.Shape.StyledPanel)
        preview_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a20;
                border: 1px solid #3a3a45;
                border-radius: 8px;
            }
        """)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(6, 6, 6, 6)

        self.screen_label = QLabel("waiting for first perception…")
        self.screen_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screen_label.setStyleSheet("color: #777; padding: 40px;")
        self.screen_label.setMinimumSize(480, 270)
        self.screen_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.screen_label.setScaledContents(False)
        preview_layout.addWidget(self.screen_label, 1)

        top_row.addWidget(preview_frame, 3)

        # Metadata column
        meta_col = QVBoxLayout()
        meta_col.setSpacing(4)

        meta_col.addWidget(self._h2("Foreground"))
        self.window_label = QLabel("—")
        self.window_label.setWordWrap(True)
        self.window_label.setStyleSheet("color: #ccc; font-family: Consolas, monospace; font-size: 9pt;")
        meta_col.addWidget(self.window_label)

        meta_col.addSpacing(8)
        meta_col.addWidget(self._h2("Elements"))
        self.element_count_label = QLabel("0 detected")
        self.element_count_label.setStyleSheet("color: #ccc; font-size: 9pt;")
        meta_col.addWidget(self.element_count_label)

        meta_col.addSpacing(8)
        meta_col.addWidget(self._h2("Target"))
        self.target_label = QLabel("(none)")
        self.target_label.setWordWrap(True)
        self.target_label.setStyleSheet("""
            color: #6fb1ff;
            font-family: Consolas, monospace;
            font-size: 9pt;
            background-color: #1a1a20;
            padding: 6px;
            border-radius: 4px;
        """)
        meta_col.addWidget(self.target_label)

        meta_col.addStretch()

        meta_widget = QWidget()
        meta_widget.setLayout(meta_col)
        meta_widget.setFixedWidth(220)
        top_row.addWidget(meta_widget)

        main.addLayout(top_row, 3)

        # Action log
        main.addWidget(self._h2("Action log"))
        self.action_log = QListWidget()
        self.action_log.setStyleSheet("""
            QListWidget {
                background-color: #1a1a20;
                border: 1px solid #3a3a45;
                border-radius: 8px;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 9pt;
                padding: 4px;
            }
            QListWidget::item { padding: 4px 6px; border-radius: 3px; }
        """)
        self.action_log.setMaximumHeight(150)
        main.addWidget(self.action_log, 1)

    @staticmethod
    def _h2(text: str) -> QLabel:
        lbl = QLabel(text)
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        lbl.setFont(f)
        lbl.setStyleSheet("color: #888; text-transform: uppercase;")
        return lbl

    # ---------------------- public slots ----------------------

    @pyqtSlot(object, object)
    def update_perception(self, screen_state, target_tag: Optional[int]):
        """Called whenever the planner has a new screen state.

        screen_state is a core.perception.ScreenState — we receive it as
        `object` to keep this widget importable without core/perception.
        """
        # Update window title
        self.window_label.setText(screen_state.foreground_window_title or "(no window)")

        # Update element count
        n = len(screen_state.elements)
        self.element_count_label.setText(f"{n} interactive element{'s' if n != 1 else ''}")

        # Update target
        if target_tag is None:
            self.target_label.setText("(no target yet)")
            self.target_label.setStyleSheet(
                "color: #777; font-family: Consolas, monospace; font-size: 9pt; "
                "background-color: #1a1a20; padding: 6px; border-radius: 4px;"
            )
        else:
            target = next((e for e in screen_state.elements if e.tag == target_tag), None)
            if target:
                meta = (
                    f"tag:    [{target.tag}]\n"
                    f"role:   {target.role}\n"
                    f"name:   {target.name or '(unnamed)'}\n"
                    f"bounds: {target.bounds}\n"
                    f"enabled: {target.is_enabled}"
                )
                self.target_label.setText(meta)
                self.target_label.setStyleSheet(
                    "color: #6fb1ff; font-family: Consolas, monospace; font-size: 9pt; "
                    "background-color: #1a1a20; padding: 6px; border-radius: 4px;"
                )
            else:
                self.target_label.setText(f"tag [{target_tag}] (not in current state)")

        # Render screen with overlays
        try:
            self._render_screen(screen_state, target_tag)
        except Exception as e:
            self.screen_label.setText(f"render error: {e}")

    @pyqtSlot(int, str, bool)
    def append_action(self, step: int, description: str, success: bool):
        """Add a row to the action log. Called from on_step_complete."""
        item = QListWidgetItem(f"[{step:2}]  {description}")
        item.setForeground(QColor("#7fdb7f") if success else QColor("#e57373"))
        self.action_log.addItem(item)
        self.action_log.scrollToBottom()
        # Keep the log bounded
        while self.action_log.count() > 200:
            self.action_log.takeItem(0)

    def reset(self):
        """Called when a new task starts."""
        self.action_log.clear()
        self.screen_label.setText("waiting for first perception…")
        self.window_label.setText("—")
        self.element_count_label.setText("0 detected")
        self.target_label.setText("(none)")

    # ---------------------- rendering ----------------------

    def _render_screen(self, screen_state, target_tag: Optional[int]):
        """Decode the base64 PNG and draw it scaled into the QLabel.

        We use the *raw* screenshot and re-render the SoM tags in PIL with
        the target highlighted. This is a tiny CPU cost (PIL drawing is fast)
        and avoids needing two separate base64 strings.
        """
        from PIL import Image
        from core.perception import VisualPerceptionEngine

        if getattr(screen_state, 'sensitive_content_detected', False):
            self._latest_pixmap = None
            self.screen_label.clear()
            self.screen_label.setText('Preview hidden: credentials detected in the target app.')
            return

        # Decode raw PNG
        raw_bytes = base64.b64decode(screen_state.raw_screenshot_b64)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        # Re-annotate with target highlighted
        engine = VisualPerceptionEngine()
        annotated = engine.annotate_with_target(pil_img, screen_state.elements, target_tag)

        # Convert PIL → QPixmap
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        qimg = QImage.fromData(buf.getvalue(), "PNG")
        pix = QPixmap.fromImage(qimg)

        # Scale to fit while maintaining aspect ratio
        scaled = pix.scaled(
            self.screen_label.width(), self.screen_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._latest_pixmap = pix
        self.screen_label.setPixmap(scaled)
        self.screen_label.setStyleSheet("")  # clear placeholder styling

    def resizeEvent(self, event):
        """Re-scale the cached pixmap on resize."""
        super().resizeEvent(event)
        if self._latest_pixmap is not None:
            scaled = self._latest_pixmap.scaled(
                self.screen_label.width(), self.screen_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.screen_label.setPixmap(scaled)
