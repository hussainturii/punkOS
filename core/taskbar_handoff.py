"""Grounded mouse handoff to an already-open supported app's taskbar button."""
import re
import time


def native_taskbar_bounds(handle):
    import ctypes
    from ctypes import wintypes
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user.GetWindowRect.restype = wintypes.BOOL
    rect = wintypes.RECT()
    if not user.GetWindowRect(handle, ctypes.byref(rect)):
        return None
    return (rect.left, rect.top, rect.right, rect.bottom)


def matching_label(name, app):
    app = {'vscode': 'Visual Studio Code', 'excel': 'Excel', 'chrome': 'Google Chrome'}.get(app, app)
    # Windows exposes either the app name or an app name plus running-window count.
    return bool(re.fullmatch(re.escape(app) + r'(?:\s*[,\-]\s*\d+\s+(?:running\s+)?windows?\.?|\s*)',
                             name.strip(), re.I))


def _reveal_taskbar(rect, x, y):
    """Hover an observed taskbar's screen edge once; never click a covering app."""
    import pyautogui
    width, height = pyautogui.size()
    if not (0 <= x < width and 0 <= y < height):
        return False
    if rect.bottom >= height - 2:
        point = (x, height - 1)
    elif rect.top <= 1:
        point = (x, 1)
    elif rect.left <= 1:
        point = (1, y)
    elif rect.right >= width - 2:
        point = (width - 1, y)
    else:
        return False
    pyautogui.moveTo(*point, duration=0.15)
    time.sleep(0.6)
    return True


def click_app_button(app, *, automation=None, click=None, diagnostic=lambda message: None, reveal=None):
    if app not in {'calculator', 'notepad', 'vscode', 'excel', 'chrome', 'start'}:
        return False
    if automation is None:
        import uiautomation as automation
    if click is None:
        import pyautogui
        click = pyautogui.click
        reveal = _reveal_taskbar
    candidates = []
    labels = []
    taskbar_count = 0
    visited = 0
    deadline = time.monotonic() + 2
    for taskbar in automation.GetRootControl().GetChildren():
        if taskbar.ClassName not in {'Shell_TrayWnd', 'Shell_SecondaryTrayWnd'}:
            continue
        taskbar_count += 1
        pending = [(taskbar, 0)]
        while pending:
            node, depth = pending.pop()
            visited += 1
            if visited > 250 or time.monotonic() > deadline:
                diagnostic('Taskbar discovery exceeded its bounded scan; no click attempted.')
                return False  # Incomplete discovery cannot establish uniqueness.
            if (node.Name or '').casefold().startswith(app):
                labels.append(f'{node.ControlTypeName}: {node.Name}; enabled={node.IsEnabled}, offscreen={node.IsOffscreen}')
            if node.ControlTypeName == 'ButtonControl' and matching_label(node.Name or '', app):
                if node.IsEnabled and not node.IsOffscreen:
                    candidates.append((node, taskbar.NativeWindowHandle))
            if depth < 8:
                pending.extend((child, depth + 1) for child in node.GetChildren())
    if len(candidates) != 1:
        diagnostic(f'Taskbars={taskbar_count}, matching visible buttons={len(candidates)}. '
                   + ('Observed app labels: ' + '; '.join(labels[:3]) if labels else 'No matching app label exposed.'))
        return False
    button, taskbar_handle = candidates[0]
    if not button.IsEnabled or button.IsOffscreen or not matching_label(button.Name or '', app):
        diagnostic('Taskbar button changed or became hidden before input.')
        return False
    rect = button.BoundingRectangle
    if rect.right <= rect.left or rect.bottom <= rect.top:
        diagnostic('Taskbar button has empty bounds.')
        return False
    x, y = (rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2
    # A taskbar provider may expose a usable point away from the rectangle's
    # center. Accept only an explicitly reported point inside fresh bounds.
    try:
        point_x, point_y, available = button.GetClickablePoint()
        if available and rect.left <= point_x < rect.right and rect.top <= point_y < rect.bottom:
            x, y = int(point_x), int(point_y)
    except (AttributeError, TypeError, ValueError):
        pass
    hit = automation.ControlFromPoint(x, y)
    hit_description = f'{hit.ControlTypeName}/{hit.ClassName}' if hit is not None else 'none'
    # Require the hit itself (or its parent) to be the selected button. Merely
    # belonging to the taskbar is insufficient when a menu/preview covers it.
    for _ in range(4):
        if hit is None:
            diagnostic('Taskbar button is covered or could not be hit-tested.')
            return False
        if automation.ControlsAreSame(hit, button):
            root = button
            for _ in range(10):
                if root is None:
                    diagnostic('The selected button lost its taskbar ancestor before input.')
                    return False
                if root.NativeWindowHandle == taskbar_handle and root.ClassName in {'Shell_TrayWnd', 'Shell_SecondaryTrayWnd'}:
                    click(x, y)
                    return True
                root = root.GetParentControl()
            diagnostic('Could not revalidate the taskbar ancestor within the scan limit.')
            return False
        hit = hit.GetParentControl()
    diagnostic(f'The button center ({x}, {y}) belongs to {hit_description}, not the selected taskbar button; '
               f'button bounds=({rect.left}, {rect.top}, {rect.right}, {rect.bottom}).')
    # Some UIA providers disagree with native point lookup across DPI contexts.
    # Require Windows' actual hit window to be this exact taskbar before falling
    # back to its freshly observed button bounds. Popup/preview HWNDs are excluded.
    native_hit = automation.ControlFromPoint2(x, y)
    if (native_hit is not None and native_hit.NativeWindowHandle == taskbar_handle
            and native_hit.ClassName in {'Shell_TrayWnd', 'Shell_SecondaryTrayWnd'}):
        click(x, y)
        return True
    if native_hit is not None:
        diagnostic(f'Native hit={native_hit.ControlTypeName}/{native_hit.ClassName}, '
                   f'point=({x},{y}), expected taskbar HWND={taskbar_handle}, '
                   f'actual HWND={native_hit.NativeWindowHandle}, taskbar bounds={native_taskbar_bounds(taskbar_handle)}.')
    # Auto-hidden taskbars can expose stale visible button bounds. Hover the
    # observed edge and repeat discovery/hit-testing once, using fresh controls.
    if reveal is not None and reveal(rect, x, y):
        return click_app_button(app, automation=automation, click=click, diagnostic=diagnostic)
    return False
