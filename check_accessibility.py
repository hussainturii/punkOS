"""Read-only app accessibility diagnostics. No keys, screenshots or model requests."""
import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', choices=['vscode', 'notepad', 'calculator', 'chrome', 'excel'], default='vscode')
    args = parser.parse_args()
    from core.window_handoff import WindowsBackend, APPS
    from core.perception import VisualPerceptionEngine, auto, UIA_AVAILABLE
    if not UIA_AVAILABLE:
        print('UI Automation is unavailable. Run in the project Windows Python environment.')
        return 1
    windows = [w for w in WindowsBackend().windows() if w.executable.casefold() in APPS[args.app]]
    if len(windows) != 1:
        print(f'Expected one {args.app} window; found {len(windows)}. No window was changed.')
        return 1
    engine = VisualPerceptionEngine()
    elements, _ = engine.get_accessibility_tree(root=auto.ControlFromHandle(windows[0].handle))
    print(f'App: {args.app}; captured controls: {len(elements)}')
    print(engine.accessibility_diagnostic())
    print('Roles:', dict(Counter(e.role for e in elements)))
    editable = sum(e.is_enabled and e.is_editable for e in elements)
    print(f'Enabled editable controls: {editable}')
    print('Focused app document is not inferred from the total control count; toolbar buttons are not text fields.')
    print('This checks accessibility only, not task completion. No input or API requests were sent.')
    return 0 if editable else 2


if __name__ == '__main__':
    raise SystemExit(main())
