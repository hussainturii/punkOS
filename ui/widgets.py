"""
Reusable UI components built on the design system in ui/theme.py.

Anything that appears in more than one panel lives here so it looks and behaves
the same everywhere (USE-7). Components read their colours from the active
palette and re-style themselves when the theme changes.

Notable pieces:
  StatusPill    animated agent state — pulsing dot / spinner (UI-4, FR1.3)
  MicButton     pulsing record indicator for voice capture (USE-6)
  ChatFeed      readable conversation transcript, not a wall of monospace
  PlanStepList  per-step status: pending / running / done / failed (FR1.4)
  RiskBadge     LOW / MEDIUM / HIGH colour coding (FR2.4)
  EmptyState    what to do when a panel has nothing in it yet
  Toast         non-blocking confirmation that something happened
"""
from __future__ import annotations

import html
import re
from typing import Optional


def result_html(message):
    """Render basic pipe tables and line breaks, escaping every user/model cell."""
    lines = message.splitlines()
    output = []
    i = 0
    while i < len(lines):
        cells = lambda line: [part.strip() for part in line.strip().strip('|').split('|')]
        if (i + 1 < len(lines) and lines[i].strip().startswith('|')
                and all(re.fullmatch(r':?-{3,}:?', cell) for cell in cells(lines[i + 1]))
                and len(cells(lines[i])) == len(cells(lines[i + 1]))):
            width = len(cells(lines[i]))
            rows = [cells(lines[i])]
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|') and len(cells(lines[i])) == width:
                rows.append(cells(lines[i]))
                i += 1
            output.append('<table border="1" cellpadding="4">' + ''.join(
                '<tr>' + ''.join('<td>' + html.escape(cell) + '</td>' for cell in row) + '</tr>'
                for row in rows) + '</table>')
        else:
            output.append(html.escape(lines[i]) + '<br>')
            i += 1
    return ''.join(output)

from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, QTimer, pyqtProperty, pyqtSignal, QSize,
)
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame,
    QListWidget, QListWidgetItem, QTextEdit, QGraphicsOpacityEffect, QSizePolicy,
)

from ui.theme import METRICS, MONO_STACK, Palette, DARK


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def set_variant(widget: QWidget, variant: str) -> QWidget:
    """Tag a widget so the global stylesheet can style it."""
    widget.setProperty("variant", variant)
    return widget


def set_role(widget: QWidget, role: str) -> QWidget:
    widget.setProperty("role", role)
    return widget


def restyle(widget: QWidget) -> None:
    """Force Qt to re-evaluate the stylesheet after a dynamic property changes.

    Qt does not repolish automatically when a property used in a selector
    changes, so toggling e.g. active="true" is a no-op without this.
    """
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def heading(text: str, level: int = 2) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", f"h{level}")
    return label


def caption(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "caption")
    label.setWordWrap(True)
    return label


def muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "muted")
    label.setWordWrap(True)
    return label


def section_label(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setProperty("role", "section")
    return label


def divider() -> QFrame:
    line = QFrame()
    line.setProperty("role", "divider")
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    return line


def card(*children: QWidget, spacing: int = METRICS.space_md,
         margins: Optional[tuple[int, int, int, int]] = None) -> QFrame:
    """A bordered surface holding a vertical stack of widgets."""
    frame = QFrame()
    frame.setProperty("role", "card")
    layout = QVBoxLayout(frame)
    m = margins or (METRICS.space_lg,) * 4
    layout.setContentsMargins(*m)
    layout.setSpacing(spacing)
    for child in children:
        layout.addWidget(child)
    return frame


# ---------------------------------------------------------------------------
# StatusPill — UI-4 / FR1.3 / USE-2
# ---------------------------------------------------------------------------

class StatusDot(QWidget):
    """A dot that pulses while the agent is busy and holds steady when idle."""

    def __init__(self, palette: Palette = DARK):
        super().__init__()
        self.palette_ = palette
        self._colour = QColor(palette.text_faint)
        self._radius = 4.0
        self._growing = True
        self.setFixedSize(14, 14)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_colour(self, colour: str) -> None:
        self._colour = QColor(colour)
        self.update()

    def set_animated(self, animated: bool) -> None:
        if animated and not self._timer.isActive():
            self._timer.start(60)
        elif not animated and self._timer.isActive():
            self._timer.stop()
            self._radius = 4.0
            self.update()

    def _tick(self) -> None:
        step = 0.22
        self._radius += step if self._growing else -step
        if self._radius >= 6.0:
            self._radius, self._growing = 6.0, False
        elif self._radius <= 3.0:
            self._radius, self._growing = 3.0, True
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Soft halo while animating, so motion reads at a glance.
        if self._timer.isActive():
            halo = QColor(self._colour)
            halo.setAlpha(60)
            painter.setBrush(halo)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(self.rect().center(), int(self._radius + 3), int(self._radius + 3))

        painter.setBrush(self._colour)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self.rect().center(), int(self._radius), int(self._radius))


class StatusPill(QFrame):
    """Agent state, always visible (USE-2).

    FR1.3 names five states: Idle, Thinking, Planning, Executing, Awaiting
    Confirmation. The planner emits free-text status strings, so we map them
    onto those five rather than inventing a sixth.
    """

    STATES = {
        "idle":       ("Idle",                  "text_faint", False),
        "thinking":   ("Thinking",              "info",       True),
        "planning":   ("Planning",              "accent",     True),
        "executing":  ("Executing",             "accent",     True),
        "awaiting":   ("Awaiting confirmation", "warning",    True),
        "completed":  ("Completed",             "success",    False),
        "failed":     ("Failed",                "danger",     False),
        "stopped":    ("Stopped",               "danger",     False),
    }

    def __init__(self, palette: Palette = DARK):
        super().__init__()
        self.palette_ = palette
        self.setProperty("role", "card")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(METRICS.space_md, METRICS.space_sm,
                                  METRICS.space_lg, METRICS.space_sm)
        layout.setSpacing(METRICS.space_sm)

        self.dot = StatusDot(palette)
        layout.addWidget(self.dot)

        self.label = QLabel("Idle")
        font = QFont()
        font.setBold(True)
        self.label.setFont(font)
        layout.addWidget(self.label)

        self.detail = QLabel("")
        self.detail.setProperty("role", "muted")
        layout.addWidget(self.detail)

        # No trailing stretch: the pill must size to its content, otherwise it
        # eats the header row and pushes the theme button off screen.
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

        self.set_state("idle")

    def set_palette(self, palette: Palette) -> None:
        self.palette_ = palette
        self.set_state(self._state_key, self.detail.text())

    def set_state(self, key: str, detail: str = "") -> None:
        key = key if key in self.STATES else "idle"
        self._state_key = key
        text, colour_token, animated = self.STATES[key]
        colour = getattr(self.palette_, colour_token)

        self.dot.set_colour(colour)
        self.dot.set_animated(animated)
        self.label.setText(text)
        self.label.setStyleSheet(f"color: {colour};")
        self.detail.setText(detail)

    def set_from_status(self, status: str) -> None:
        """Map a planner status string onto one of the five FR1.3 states."""
        low = status.lower()
        detail = status

        if "perceiv" in low or "think" in low:
            key = "thinking"
        elif "plan" in low:
            key = "planning"
        elif "execut" in low or "sandbox" in low or "web agent" in low:
            key = "executing"
        elif "awaiting" in low or "confirm" in low:
            key = "awaiting"
        elif "complet" in low:
            key, detail = "completed", ""
        elif "fail" in low:
            key, detail = "failed", ""
        elif "stop" in low:
            key, detail = "stopped", ""
        else:
            key, detail = "idle", ""

        # The status already carries the human-readable phrase; only show it as
        # detail when it adds information beyond the state name itself.
        state_name = self.STATES[key][0].lower()
        if detail.lower().strip() == state_name:
            detail = ""
        self.set_state(key, detail)


# ---------------------------------------------------------------------------
# RiskBadge — FR2.4
# ---------------------------------------------------------------------------

class RiskBadge(QLabel):
    def __init__(self, level: str = "LOW", palette: Palette = DARK):
        super().__init__()
        self.palette_ = palette
        self.set_level(level)

    def set_level(self, level: str) -> None:
        level = (level or "LOW").upper()
        colour = {
            "LOW": self.palette_.risk_low,
            "MEDIUM": self.palette_.risk_medium,
            "HIGH": self.palette_.risk_high,
        }.get(level, self.palette_.risk_low)

        self.setText(f"  {level} RISK  ")
        self.setStyleSheet(
            f"color: {self.palette_.text_on_accent};"
            f"background-color: {colour};"
            f"border-radius: {METRICS.radius_pill}px;"
            f"padding: 3px 10px; font-weight: 700; font-size: {METRICS.font_sm}pt;"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)


# ---------------------------------------------------------------------------
# MicButton — USE-6
# ---------------------------------------------------------------------------

class MicButton(QPushButton):
    """Microphone toggle that visibly pulses while capturing audio."""

    IDLE_GLYPH = "●"       # geometric glyphs render in the UI font; the 🎤
    RECORDING_GLYPH = "■"  # emoji falls back to a font with different metrics

    def __init__(self, palette: Palette = DARK):
        super().__init__(self.IDLE_GLYPH)
        self.palette_ = palette
        self.setToolTip("Record a voice command  (Ctrl+M)")
        self.setFixedSize(METRICS.control_height + 6, METRICS.control_height + 6)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._recording = False
        self._phase = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pulse)
        self._apply_style(False)

    def set_palette(self, palette: Palette) -> None:
        self.palette_ = palette
        self._apply_style(self._recording)

    def set_recording(self, recording: bool) -> None:
        self._recording = recording
        if recording:
            self.setText(self.RECORDING_GLYPH)
            self.setToolTip("Stop recording and transcribe")
            self._timer.start(90)
        else:
            self._timer.stop()
            self.setText(self.IDLE_GLYPH)
            self.setToolTip("Record a voice command  (Ctrl+M)")
        self._apply_style(recording)

    def _pulse(self) -> None:
        self._phase = (self._phase + 1) % 8
        # Vary the ring opacity so the user can see capture is live.
        alpha = 40 + abs(4 - self._phase) * 22
        self.setStyleSheet(
            f"QPushButton {{"
            f" background-color: {self.palette_.danger};"
            f" color: white; border: 3px solid rgba(248,113,113,{alpha / 255:.2f});"
            f" border-radius: {METRICS.radius_md}px; font-size: {METRICS.font_lg}pt; }}"
        )

    def _apply_style(self, recording: bool) -> None:
        if recording:
            self._pulse()
            return
        self.setStyleSheet(
            f"QPushButton {{"
            f" background-color: {self.palette_.surface_alt};"
            f" color: {self.palette_.text};"
            f" border: 1px solid {self.palette_.border};"
            f" border-radius: {METRICS.radius_md}px;"
            f" font-size: {METRICS.font_lg}pt; }}"
            f"QPushButton:hover {{ background-color: {self.palette_.surface_hover};"
            f" border-color: {self.palette_.accent}; }}"
        )


# ---------------------------------------------------------------------------
# ChatFeed
# ---------------------------------------------------------------------------

class ChatFeed(QTextEdit):
    """The conversation transcript.

    The old version dumped every planner log line into one monospace blob. This
    keeps the agent's technical log readable but gives the user's own commands
    and the final results real visual weight, so you can scan a run's outcome
    without reading every line.
    """

    def __init__(self, palette: Palette = DARK):
        super().__init__()
        self.palette_ = palette
        self.setReadOnly(True)
        self.document().setDocumentMargin(METRICS.space_lg)
        # Messages are kept as structured data, not just rendered HTML. Colours
        # are baked into the HTML at insert time, so a theme switch has to
        # regenerate the whole transcript — otherwise every message already on
        # screen keeps the old palette and turns unreadable.
        self._messages: list[tuple[str, tuple]] = []
        self._apply_style()

    def set_palette(self, palette: Palette) -> None:
        self.palette_ = palette
        self._apply_style()
        self._rerender()

    def _rerender(self) -> None:
        replayed = list(self._messages)
        self.clear()
        self._messages = []
        for kind, args in replayed:
            getattr(self, kind)(*args)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"QTextEdit {{ background-color: {self.palette_.surface};"
            f" border: 1px solid {self.palette_.border};"
            f" border-radius: {METRICS.radius_lg}px;"
            f" font-size: {METRICS.font_md}pt; }}"
        )

    # -- message kinds ------------------------------------------------

    def add_user_message(self, text: str) -> None:
        self._messages.append(("add_user_message", (text,)))
        self._append(
            f'<div style="color:{self.palette_.accent}; font-weight:700;'
            f' font-size:{METRICS.font_sm}pt;">YOU</div>'
        )
        self._append(
            f'<div style="color:{self.palette_.text};'
            f' font-size:{METRICS.font_lg}pt;">{html.escape(text)}</div>'
        )

    def add_agent_log(self, text: str) -> None:
        """A raw planner log line. Dim, monospace, secondary."""
        stripped = text.strip()
        if not stripped or stripped in {"=" * 60, "-" * 60}:
            return
        self._messages.append(("add_agent_log", (text,)))
        colour = self.palette_.text_muted
        if stripped.startswith(("❌", "✗")) or "error" in stripped.lower():
            colour = self.palette_.danger
        elif stripped.startswith(("✅", "✓")):
            colour = self.palette_.success
        elif stripped.startswith("⚠"):
            colour = self.palette_.warning

        # Planner logs arrive as multi-line blocks; each line becomes its own
        # paragraph so nothing runs together.
        for line in text.split("\n"):
            if not line.strip():
                continue
            self._append(
                f'<div style="color:{colour}; font-family:{MONO_STACK};'
                f' font-size:{METRICS.font_sm}pt;">'
                f'{html.escape(line)}</div>'
            )

    def add_result(self, success: bool, message: str, meta: str = "") -> None:
        self._messages.append(("add_result", (success, message, meta)))
        colour = self.palette_.success if success else self.palette_.danger
        icon = "✓" if success else "✗"
        self._append(
            f'<div style="background-color:{self.palette_.surface_alt};'
            f' border-left:3px solid {colour}; padding:8px;">'
            f'<span style="color:{colour}; font-weight:700;">'
            f'{icon}</span><div>{result_html(message)}</div>'
            + (f'<br><span style="color:{self.palette_.text_faint};'
               f' font-size:{METRICS.font_sm}pt;">{html.escape(meta)}</span>' if meta else "")
            + '</div>'
        )

    def add_system_note(self, text: str, kind: str = "info") -> None:
        self._messages.append(("add_system_note", (text, kind)))
        colour = {
            "info": self.palette_.info,
            "warning": self.palette_.warning,
            "danger": self.palette_.danger,
        }.get(kind, self.palette_.info)
        self._append(
            f'<div style="color:{colour}; font-size:{METRICS.font_sm}pt;">'
            f'{html.escape(text)}</div>'
        )

    def _append(self, html_fragment: str) -> None:
        """Append as a NEW paragraph.

        insertHtml() merges consecutive <div>s into the current block, which ran
        every log line together into one wall of text. append() starts a fresh
        block each time, which is what we actually want here.
        """
        self.append(html_fragment)
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_feed(self) -> None:
        self._messages = []
        self.clear()


# ---------------------------------------------------------------------------
# PlanStepList — FR1.4
# ---------------------------------------------------------------------------

class PlanStepList(QListWidget):
    """Ordered sub-tasks with per-step status icons."""

    ICONS = {
        "pending": "○",
        "running": "◐",
        "done": "●",
        "failed": "✕",
    }

    def __init__(self, palette: Palette = DARK):
        super().__init__()
        self.palette_ = palette
        self._steps: list[dict] = []
        self.setWordWrap(True)
        self.setSpacing(2)

    def set_palette(self, palette: Palette) -> None:
        self.palette_ = palette
        self._render()

    def set_steps(self, steps: list[dict]) -> None:
        self._steps = [
            {
                "number": s.get("step_number", i + 1),
                "description": s.get("description", ""),
                "high_risk": bool(s.get("is_high_risk")),
                "status": "pending",
            }
            for i, s in enumerate(steps)
        ]
        self._render()

    def mark(self, index: int, status: str) -> None:
        if 0 <= index < len(self._steps):
            self._steps[index]["status"] = status
            self._render()

    def mark_current(self, index: int) -> None:
        for i, step in enumerate(self._steps):
            if step["status"] in ("done", "failed"):
                continue
            step["status"] = "running" if i == index else "pending"
        self._render()

    def clear_steps(self) -> None:
        self._steps = []
        self.clear()

    def _render(self) -> None:
        self.clear()
        colours = {
            "pending": self.palette_.text_faint,
            "running": self.palette_.accent,
            "done": self.palette_.success,
            "failed": self.palette_.danger,
        }
        for step in self._steps:
            icon = self.ICONS[step["status"]]
            risk = "  ⚠" if step["high_risk"] else ""
            item = QListWidgetItem(f" {icon}  {step['number']}. {step['description']}{risk}")
            item.setForeground(QColor(colours[step["status"]]))
            if step["status"] == "running":
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.addItem(item)


# ---------------------------------------------------------------------------
# EmptyState
# ---------------------------------------------------------------------------

class EmptyState(QWidget):
    """Shown when a panel has no content — says what to do, not just 'no data'."""

    def __init__(self, icon: str, title: str, hint: str, palette: Palette = DARK):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(METRICS.space_sm)

        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet(f"font-size: 40pt; color: {palette.text_faint};")
        layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            f"color: {palette.text}; font-size: {METRICS.font_lg}pt; font-weight: 600;"
        )
        layout.addWidget(title_label)

        hint_label = QLabel(hint)
        hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet(f"color: {palette.text_muted};")
        layout.addWidget(hint_label)


# ---------------------------------------------------------------------------
# Toast
# ---------------------------------------------------------------------------

class Toast(QLabel):
    """Brief non-blocking confirmation, fades out on its own.

    Used for things the user should notice but not have to dismiss — "settings
    saved", "index rebuilt". A modal for those would be user-hostile.
    """

    def __init__(self, parent: QWidget, palette: Palette = DARK):
        super().__init__(parent)
        self.palette_ = palette
        self.setVisible(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._anim = QPropertyAnimation(self._effect, b"opacity", self)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out)

    def show_message(self, text: str, kind: str = "info", duration_ms: int = 2600) -> None:
        colour = {
            "info": self.palette_.accent,
            "success": self.palette_.success,
            "warning": self.palette_.warning,
            "danger": self.palette_.danger,
        }.get(kind, self.palette_.accent)

        self.setText(f"  {text}  ")
        self.setStyleSheet(
            f"background-color: {self.palette_.surface_alt};"
            f"color: {self.palette_.text};"
            f"border: 1px solid {colour};"
            f"border-left: 3px solid {colour};"
            f"border-radius: {METRICS.radius_md}px;"
            f"padding: {METRICS.space_md}px {METRICS.space_lg}px;"
        )
        self.adjustSize()
        self._reposition()
        self.setVisible(True)
        self.raise_()

        self._anim.stop()
        self._anim.setDuration(160)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()
        self._hide_timer.start(duration_ms)

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        x = (parent.width() - self.width()) // 2
        self.move(max(0, x), METRICS.space_xl)

    def _fade_out(self) -> None:
        self._anim.stop()
        self._anim.setDuration(280)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.finished.connect(self._on_faded)
        self._anim.start()

    def _on_faded(self) -> None:
        try:
            self._anim.finished.disconnect(self._on_faded)
        except TypeError:
            pass
        self.setVisible(False)
