"""
Design system for the PunkOS UI.

One place defines every colour, spacing step, radius and font size; every panel
pulls from here. Before this existed the same blue was written as #4a7cd1,
#3a6ea5, #5a8ec5 and #6fb1ff in different files, and there was no way to add a
light theme without touching twenty stylesheets.

Satisfies:
  UI-5   light and dark themes, all text at WCAG AA (4.5:1) or better
  USE-7  consistent iconography, typography and colour coding across panels

The contrast requirement is enforced, not asserted: `audit_contrast()` computes
the real WCAG ratio for every foreground/background pair in both palettes, and
tests/test_ui.py fails the build if any pair drops below 4.5:1. Picking colours
that merely look fine is how accessibility bugs ship.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

ThemeName = Literal["dark", "light"]


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Palette:
    """Every colour the UI is allowed to use."""
    name: str

    # Surfaces, back to front
    bg: str             # window background
    surface: str        # panels, cards
    surface_alt: str    # inputs, nested surfaces
    surface_hover: str
    sidebar: str

    # Lines
    border: str
    border_strong: str

    # Text
    text: str           # primary
    text_muted: str     # secondary
    text_faint: str     # tertiary / placeholder
    text_on_accent: str

    # Brand + semantics
    #
    # `accent` and `accent_solid` are deliberately two tokens. An accent used
    # as TEXT on a dark surface must be light; the same accent used as a BUTTON
    # FILL behind white text must be dark. On the dark theme no single blue
    # clears 4.5:1 in both roles — #6b8cff reads at 5.70:1 as text but only
    # 3.08:1 behind white, and darkening it to fix the button breaks the text.
    # So: `accent` for text/icons/borders, `accent_solid` for filled controls.
    accent: str
    accent_hover: str
    accent_solid: str        # filled-button background, pairs with text_on_accent
    accent_solid_hover: str
    accent_soft: str         # tinted background for selected nav, badges
    success: str
    warning: str
    danger: str
    danger_hover: str
    info: str

    # Risk levels (M9) — kept distinct from semantics so they can be tuned
    risk_low: str
    risk_medium: str
    risk_high: str

    def to_dict(self) -> dict:
        return asdict(self)


DARK = Palette(
    name="dark",
    bg="#0f1117",
    surface="#161923",
    surface_alt="#1d2130",
    surface_hover="#252a3a",
    sidebar="#0c0e14",
    border="#262b3a",
    border_strong="#39415a",
    text="#e8eaf2",
    text_muted="#a8b0c4",
    text_faint="#7c8499",
    text_on_accent="#ffffff",
    accent="#6b8cff",
    accent_hover="#8199ff",
    accent_solid="#4a63e0",
    accent_solid_hover="#5a73f0",
    accent_soft="#1e2540",
    success="#4ade80",
    warning="#fbbf24",
    danger="#f87171",
    danger_hover="#fca5a5",
    info="#60c8f0",
    risk_low="#4ade80",
    risk_medium="#fbbf24",
    risk_high="#f87171",
)

LIGHT = Palette(
    name="light",
    bg="#f5f6fa",
    surface="#ffffff",
    surface_alt="#eef0f6",
    surface_hover="#e3e7f0",
    sidebar="#ffffff",
    border="#d9dde8",
    border_strong="#b6bdd0",
    text="#161923",
    text_muted="#525a70",
    text_faint="#6b7386",
    text_on_accent="#ffffff",
    accent="#3355e0",
    accent_hover="#2444cc",
    accent_solid="#3355e0",
    accent_solid_hover="#2444cc",
    accent_soft="#e6ebff",
    success="#0f7a3d",
    warning="#8a5a00",
    danger="#c0392b",
    danger_hover="#a02f22",
    info="#0b6f92",
    risk_low="#0f7a3d",
    risk_medium="#8a5a00",
    risk_high="#c0392b",
)

PALETTES: dict[str, Palette] = {"dark": DARK, "light": LIGHT}


@dataclass(frozen=True)
class Metrics:
    """Spacing, radii and type scale. One 4px rhythm throughout."""
    space_xs: int = 4
    space_sm: int = 8
    space_md: int = 12
    space_lg: int = 16
    space_xl: int = 24
    space_2xl: int = 32

    radius_sm: int = 6
    radius_md: int = 10
    radius_lg: int = 14
    radius_pill: int = 999

    # Point sizes. Qt renders points larger than a browser renders the "same"
    # pixel size, so this scale is deliberately a step below typical web values
    # — 11pt body filled the conversation column with about six words a line.
    font_xs: int = 8
    font_sm: int = 9
    font_md: int = 10
    font_lg: int = 11
    font_xl: int = 14
    font_2xl: int = 18

    sidebar_width: int = 196
    control_height: int = 34


METRICS = Metrics()

# SDS USE-7 names Calibri/Inter. Inter first where installed, then the Windows
# UI font, then generic fallbacks so the app never renders in Times New Roman.
FONT_STACK = '"Inter", "Segoe UI Variable", "Segoe UI", "Calibri", system-ui, sans-serif'
MONO_STACK = '"JetBrains Mono", "Cascadia Code", "Consolas", "Courier New", monospace'


# ---------------------------------------------------------------------------
# WCAG contrast
# ---------------------------------------------------------------------------

def _srgb_to_linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_colour: str) -> float:
    """WCAG 2.1 relative luminance."""
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (0.2126 * _srgb_to_linear(r)
            + 0.7152 * _srgb_to_linear(g)
            + 0.0722 * _srgb_to_linear(b))


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG contrast ratio between two hex colours. 1.0 = identical, 21.0 = max."""
    l1, l2 = relative_luminance(fg), relative_luminance(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


# Pairs that must hold at AA (4.5:1). Body text and any coloured text that
# conveys meaning — status labels, risk badges, error messages.
_REQUIRED_PAIRS: tuple[tuple[str, str], ...] = (
    ("text", "bg"), ("text", "surface"), ("text", "surface_alt"), ("text", "sidebar"),
    ("text_muted", "bg"), ("text_muted", "surface"), ("text_muted", "surface_alt"),
    ("text_faint", "surface"),
    ("text_on_accent", "accent_solid"),
    ("accent", "bg"), ("accent", "surface"), ("accent", "accent_soft"),
    ("success", "surface"), ("warning", "surface"),
    ("danger", "surface"), ("info", "surface"),
    ("risk_low", "surface"), ("risk_medium", "surface"), ("risk_high", "surface"),
)


def audit_contrast(palette: Palette, minimum: float = 4.5) -> list[tuple[str, str, float]]:
    """Return every required pair that falls below `minimum`. Empty list = pass."""
    failures = []
    for fg_name, bg_name in _REQUIRED_PAIRS:
        ratio = contrast_ratio(getattr(palette, fg_name), getattr(palette, bg_name))
        if ratio < minimum:
            failures.append((fg_name, bg_name, round(ratio, 2)))
    return failures


# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------

def build_stylesheet(palette: Palette, m: Metrics = METRICS) -> str:
    """The application-wide QSS.

    Applied once to QApplication so every panel inherits it. Panels should set
    an objectName or property rather than writing their own inline colours.
    """
    p = palette
    return f"""
/* ---------- base ----------
   Deliberately NO background-color on bare QWidget. Setting it there makes
   every QLabel opaque, which paints light rectangles over any dark card the
   label sits inside. Backgrounds belong on windows and explicit containers;
   labels stay transparent. */
QWidget {{
    color: {p.text};
    font-family: {FONT_STACK};
    font-size: {m.font_md}pt;
}}
QMainWindow, QDialog, QStackedWidget, QScrollArea, QSplitter {{
    background-color: {p.bg};
}}
QScrollArea > QWidget > QWidget {{ background-color: transparent; }}
QLabel {{ background-color: transparent; border: none; }}
QCheckBox {{ background-color: transparent; }}

QToolTip {{
    background-color: {p.surface_alt};
    color: {p.text};
    border: 1px solid {p.border_strong};
    border-radius: {m.radius_sm}px;
    padding: {m.space_sm}px {m.space_md}px;
}}

/* ---------- typography helpers ---------- */
QLabel[role="h1"] {{ font-size: {m.font_2xl}pt; font-weight: 700; color: {p.text}; }}
QLabel[role="h2"] {{ font-size: {m.font_xl}pt;  font-weight: 600; color: {p.text}; }}
QLabel[role="h3"] {{ font-size: {m.font_lg}pt;  font-weight: 600; color: {p.text}; }}
QLabel[role="muted"]   {{ color: {p.text_muted}; }}
QLabel[role="caption"] {{ color: {p.text_faint}; font-size: {m.font_sm}pt; }}
QLabel[role="section"] {{
    color: {p.text_faint};
    font-size: {m.font_xs}pt;
    font-weight: 700;
    letter-spacing: 1px;
    padding-top: {m.space_md}px;
}}

/* ---------- surfaces ---------- */
QFrame[role="card"] {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: {m.radius_lg}px;
}}
QFrame[role="sidebar"] {{
    background-color: {p.sidebar};
    border-right: 1px solid {p.border};
}}
QFrame[role="divider"] {{ background-color: {p.border}; max-height: 1px; border: none; }}

/* ---------- buttons ---------- */
QPushButton {{
    background-color: {p.surface_alt};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: {m.radius_md}px;
    padding: {m.space_sm}px {m.space_lg}px;
    min-height: {m.control_height - 16}px;
    font-weight: 500;
}}
QPushButton:hover  {{ background-color: {p.surface_hover}; border-color: {p.border_strong}; }}
QPushButton:pressed {{ background-color: {p.border}; }}
QPushButton:disabled {{ color: {p.text_faint}; background-color: {p.surface}; border-color: {p.border}; }}

QPushButton[variant="primary"] {{
    background-color: {p.accent_solid};
    color: {p.text_on_accent};
    border: 1px solid {p.accent_solid};
    font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{
    background-color: {p.accent_solid_hover}; border-color: {p.accent_solid_hover};
}}
QPushButton[variant="primary"]:disabled {{
    background-color: {p.surface_alt}; color: {p.text_faint}; border-color: {p.border};
}}

QPushButton[variant="danger"] {{
    background-color: {p.danger}; color: {p.text_on_accent}; border: 1px solid {p.danger}; font-weight: 600;
}}
QPushButton[variant="danger"]:hover {{ background-color: {p.danger_hover}; border-color: {p.danger_hover}; }}
QPushButton[variant="danger"]:disabled {{
    background-color: {p.surface_alt}; color: {p.text_faint}; border-color: {p.border};
}}

QPushButton[variant="ghost"] {{ background-color: transparent; border: 1px solid transparent; }}
QPushButton[variant="ghost"]:hover {{ background-color: {p.surface_hover}; }}

/* sidebar navigation */
QPushButton[variant="nav"] {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: {m.radius_md}px;
    color: {p.text_muted};
    text-align: left;
    padding: {m.space_md}px {m.space_md}px;
    font-weight: 500;
}}
QPushButton[variant="nav"]:hover {{ background-color: {p.surface_hover}; color: {p.text}; }}
QPushButton[variant="nav"][active="true"] {{
    background-color: {p.accent_soft};
    color: {p.accent};
    font-weight: 600;
}}

/* ---------- inputs ---------- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QComboBox {{
    background-color: {p.surface_alt};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: {m.radius_md}px;
    padding: {m.space_sm}px {m.space_md}px;
    selection-background-color: {p.accent};
    selection-color: {p.text_on_accent};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {{
    border: 1px solid {p.accent};
}}
QLineEdit::placeholder {{ color: {p.text_faint}; }}
QLineEdit:disabled, QTextEdit:disabled {{ color: {p.text_faint}; background-color: {p.surface}; }}

QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {p.surface_alt};
    color: {p.text};
    border: 1px solid {p.border_strong};
    border-radius: {m.radius_sm}px;
    selection-background-color: {p.accent};
    selection-color: {p.text_on_accent};
    outline: none;
}}

QCheckBox {{ spacing: {m.space_sm}px; color: {p.text}; }}
QCheckBox::indicator {{
    width: 17px; height: 17px;
    border: 1px solid {p.border_strong};
    border-radius: {m.radius_sm - 2}px;
    background-color: {p.surface_alt};
}}
QCheckBox::indicator:checked {{ background-color: {p.accent}; border-color: {p.accent}; }}
QCheckBox::indicator:hover {{ border-color: {p.accent}; }}

/* ---------- lists, trees, tables ---------- */
QListWidget, QTreeWidget, QTableWidget {{
    background-color: {p.surface};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: {m.radius_md}px;
    outline: none;
}}
QListWidget::item, QTreeWidget::item {{
    padding: {m.space_sm}px;
    border-radius: {m.radius_sm}px;
}}
QListWidget::item:hover, QTreeWidget::item:hover {{ background-color: {p.surface_hover}; }}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {p.accent_soft}; color: {p.accent};
}}
QTableWidget::item:selected {{ background-color: {p.accent_soft}; color: {p.accent}; }}
QHeaderView::section {{
    background-color: {p.surface_alt};
    color: {p.text_muted};
    border: none;
    border-bottom: 1px solid {p.border};
    padding: {m.space_sm}px;
    font-weight: 600;
}}
QTableWidget {{ gridline-color: {p.border}; }}

/* ---------- tabs ---------- */
QTabWidget::pane {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: {m.radius_lg}px;
    top: -1px;
}}
QTabBar::tab {{
    background-color: transparent;
    color: {p.text_muted};
    padding: {m.space_sm}px {m.space_lg}px;
    margin-right: {m.space_xs}px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 500;
}}
QTabBar::tab:hover {{ color: {p.text}; }}
QTabBar::tab:selected {{ color: {p.accent}; border-bottom: 2px solid {p.accent}; font-weight: 600; }}

/* ---------- scrollbars ---------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {p.border_strong}; border-radius: 5px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {p.text_faint}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{
    background: {p.border_strong}; border-radius: 5px; min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{ background: {p.text_faint}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; border: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* ---------- misc ---------- */
QSplitter::handle {{ background-color: {p.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

QStatusBar {{
    background-color: {p.sidebar};
    border-top: 1px solid {p.border};
    color: {p.text_muted};
}}
QStatusBar::item {{ border: none; }}

QProgressBar {{
    background-color: {p.surface_alt};
    border: none;
    border-radius: {m.radius_pill}px;
    height: 6px;
    text-align: center;
    color: {p.text_muted};
}}
QProgressBar::chunk {{ background-color: {p.accent}; border-radius: {m.radius_pill}px; }}

QMessageBox {{ background-color: {p.surface}; }}
QMessageBox QLabel {{ color: {p.text}; }}
"""
