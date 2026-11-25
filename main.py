"""
PunkOS — Main entry point (GUI version).

Usage:
    python main.py

Why this file constructs QApplication before importing anything else
--------------------------------------------------------------------
Windows lets a process set its DPI awareness exactly once; every later call
returns ACCESS_DENIED. Two things in PunkOS want to set it:

  * `uiautomation` calls SetProcessDpiAwareness(PerMonitorDpiAware) at *import*
    time — merely importing core.perception is enough to trigger it.
  * Qt sets PER_MONITOR_AWARE_V2 when QApplication is constructed.

Whoever runs first wins, and the loser prints a warning. Importing the
dashboard first meant uiautomation always won, and Qt logged:

    qt.qpa.window: SetProcessDpiAwarenessContext() failed: Access is denied.

So QApplication is created here, before the `ui.dashboard` import pulls in
uiautomation. Qt gets V2 (crisper UI, correct behaviour when windows move
between monitors of different scaling) and uiautomation's later call fails
silently, which it already tolerates.

This is not cosmetic. The process must be DPI-aware for the agent to click
accurately: a DPI-*unaware* process is fed virtualised 96-DPI coordinates by
Windows, while the accessibility tree reports real physical pixels. On a 125%
or 150% display those disagree, and every click lands off-target. Either
awareness mode is fine as long as one of them is actually set — what matters
is that the process is aware, and that only one component tries to set it.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root on sys.path before any local imports.
sys.path.insert(0, str(Path(__file__).parent))

# Qt first — see the module docstring. Keep this above the ui.dashboard import.
from PyQt6.QtWidgets import QApplication

_app = QApplication(sys.argv)
_app.setStyle("Fusion")
_app.setApplicationName("PunkOS")

from ui.dashboard import main  # noqa: E402  (deliberately after QApplication)

if __name__ == "__main__":
    main()
