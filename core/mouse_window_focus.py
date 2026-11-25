"""Mouse focus from observed, unobstructed title-bar controls."""
import os


def minimize_own_dashboard(*, automation=None, click=None, process_id=None):
    """Click only this process's dashboard Minimize button, verified at the point."""
    if automation is None:
        import uiautomation as automation
    if click is None:
        import pyautogui
        click = pyautogui.click
    foreground = automation.GetForegroundControl()
    if (foreground is None or foreground.ProcessId != (process_id or os.getpid())
            or foreground.Name != 'PunkOS — Agentic Desktop AI'):
        return False
    bars = [c for c in foreground.GetChildren() if c.ControlTypeName == 'TitleBarControl']
    buttons = [c for bar in bars for c in bar.GetChildren()
               if c.ControlTypeName == 'ButtonControl' and c.Name == 'Minimize'
               and c.IsEnabled and not c.IsOffscreen]
    if len(buttons) != 1:
        return False
    button = buttons[0]
    rect = button.BoundingRectangle
    if rect.right <= rect.left or rect.bottom <= rect.top:
        return False
    x, y = int((rect.left+rect.right)/2), int((rect.top+rect.bottom)/2)
    hit = automation.ControlFromPoint(x, y)
    if hit is None or not automation.ControlsAreSame(hit, button):
        return False
    click(x, y)
    return True


def click_titlebar(handle, *, automation=None, click=None):
    if automation is None:
        import uiautomation as automation
    if click is None:
        import pyautogui
        click = pyautogui.click
    window = automation.ControlFromHandle(handle)
    if window is None:
        return False
    pending = [(window, 0)]
    bars = []
    visited = 0
    while pending:
        node, depth = pending.pop()
        visited += 1
        if visited > 80:
            return False
        if node.ControlTypeName == 'TitleBarControl':
            if node.IsEnabled and not node.IsOffscreen:
                bars.append(node)
        elif depth < 2:
            pending.extend((child, depth + 1) for child in node.GetChildren())
    if len(bars) != 1:
        return False
    bar = bars[0]
    rect = bar.BoundingRectangle
    if rect.right <= rect.left or rect.bottom <= rect.top:
        return False
    for fraction in (0.5, 0.3):
        x = int(rect.left + (rect.right-rect.left) * fraction)
        y = int((rect.top+rect.bottom)/2)
        hit = automation.ControlFromPoint(x, y)
        # Do not click title-bar buttons, menus, or tabs. Only the bar itself
        # or its inert text can be used to focus without changing app state.
        if hit is None or hit.ControlTypeName not in {'TitleBarControl', 'TextControl'}:
            continue
        for _ in range(3):
            if hit is None:
                break
            if automation.ControlsAreSame(hit, bar):
                click(x, y)
                return True
            hit = hit.GetParentControl()
    return False
