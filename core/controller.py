"""
M5: Desktop Control Subsystem
The "hands" — executes actions via PyAutoGUI, grounded in accessibility tree
element bounds rather than raw pixel coordinates.
"""
from __future__ import annotations

import time
import re
from dataclasses import dataclass

import pyautogui

from .perception import ScreenState, UIElement
from .reasoning import NextAction


# Safety: moving the mouse to (0,0) aborts PyAutoGUI immediately.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1   # Small pause between pyautogui calls for stability


@dataclass
class ExecutionResult:
    success: bool
    message: str
    input_unverified: bool = False


class DesktopController:
    """Translates a NextAction into actual OS-level mouse/keyboard input."""

    def __init__(self, window_backend=None):
        self.window_backend = window_backend
        self.focus_restore_detail = ''

    def _windows(self):
        if self.window_backend is None:
            from .window_handoff import WindowsBackend
            self.window_backend = WindowsBackend()
        return self.window_backend

    def restore_focus(self, state):
        """Restore only the exact observed window after our confirmation dialog."""
        handle = state.foreground_window_handle
        self.focus_restore_detail = ''
        if not handle:
            self.focus_restore_detail = 'Observation has no window handle.'
            return False
        backend = self._windows()
        try:
            backend.activate(handle)
            actual = backend.foreground()
        except Exception as exc:
            # Record the error category without leaking window titles or content.
            self.focus_restore_detail = f'Window activation/check raised {type(exc).__name__}.'
            return False
        if not actual:
            self.focus_restore_detail = f'No foreground window identified after activating handle {handle}.'
            return False
        if actual.handle != handle:
            self.focus_restore_detail = f'Expected window handle {handle}; foreground handle is {actual.handle}.'
            return False
        return True

    def execute(self, action: NextAction, screen_state: ScreenState) -> ExecutionResult:
        try:
            if screen_state.foreground_window_handle and action.action_type not in ('done', 'fail', 'wait'):
                actual = self._windows().foreground()
                if not actual or actual.handle != screen_state.foreground_window_handle:
                    return ExecutionResult(False, 'Foreground window changed since observation; no input sent. Reobserve before retrying.')
            if action.action_type == "click":
                return self._click(action, screen_state)
            if action.action_type == "double_click":
                return self._click(action, screen_state, double=True)
            if action.action_type == "right_click":
                return self._click(action, screen_state, button="right")
            if action.action_type == "type":
                return self._type(action, screen_state)
            if action.action_type == "drag":
                return self._drag_and_drop(action, screen_state)
            if action.action_type == "key":
                return self._key(action.text)
            if action.action_type == "scroll":
                return self._scroll(action, screen_state)
            if action.action_type == "wait":
                time.sleep(1.0)
                return ExecutionResult(True, "Waited 1s")
            if action.action_type == "done":
                return ExecutionResult(True, f"Task complete: {action.text}")
            if action.action_type == "fail":
                return ExecutionResult(False, f"Agent gave up: {action.text}")
            return ExecutionResult(False, f"Unknown action type: {action.action_type}")
        except Exception as e:
            return ExecutionResult(False, f"Execution error: {e}")

    @staticmethod
    def _find_element(tag: int, screen_state: ScreenState) -> UIElement | None:
        for el in screen_state.elements:
            if el.tag == tag:
                return el
        return None

    def _click(
        self,
        action: NextAction,
        screen_state: ScreenState,
        double: bool = False,
        button: str = "left",
    ) -> ExecutionResult:
        if action.target_tag is None:
            return ExecutionResult(False, f"{action.action_type} requires target_tag")
        el = self._find_element(action.target_tag, screen_state)
        if el is not None and el.focus_only:
            return ExecutionResult(False, 'This editor has no clickable rectangle. Use a targeted type action to focus its accessibility control.')
        if el is None:
            return ExecutionResult(
                False, f"Tag {action.target_tag} not in current screen state ({len(screen_state.elements)} elements)"
            )
        if not el.is_enabled:
            return ExecutionResult(False, f"Element [{el.tag}] '{el.name}' is disabled")
        if not el.is_interactive:
            # Read-only context elements (static labels, images) are in the list
            # so the model can *read* them, not click them. Clicking one is
            # almost always the model misreading the element list.
            return ExecutionResult(
                False,
                f"Element [{el.tag}] '{el.name}' is read-only ({el.role}) and cannot be clicked"
            )
        x, y = el.center
        if double:
            pyautogui.doubleClick(x, y, button=button)
            verb = "Double-clicked"
        else:
            pyautogui.click(x, y, button=button)
            verb = f"{button.capitalize()}-clicked"
        return ExecutionResult(True, f"{verb} [{el.tag}] {el.role} '{el.name}' at ({x},{y})")

    def _type(self, action: NextAction, screen_state: ScreenState) -> ExecutionResult:
        """Type text into the focused element.

        If a target_tag is given, click that element first to guarantee it has
        focus — this prevents typing into the wrong window (e.g. PunkOS's own
        input field instead of the Run dialog).
        """
        text = action.text
        if not text:
            return ExecutionResult(False, "type action requires non-empty text")
        # Navigation text is not a keyboard command. Browsers may discard a
        # pasted trailing newline, producing a false exact-readback mismatch.
        # Reject before focus/select-all/paste; never silently change requested
        # document text or turn a newline into an implicit Enter press.
        url_like = re.fullmatch(r'(?:https?://[^\s]+|www\.[^\s]+|[\w-]+(?:\.[\w-]+)+(?::\d+)?(?:/[^\s]*)?)',
                                text.rstrip('\r\n'), re.I)
        target = self._find_element(action.target_tag, screen_state) if action.target_tag is not None else None
        navigation_field = target and target.role in {'EditControl', 'ComboBoxControl'} and re.search(
            r'\b(?:address|location|url)\b', target.name, re.I)
        if navigation_field and url_like and text.endswith(('\r', '\n')):
            return ExecutionResult(False, 'Navigation text contains a trailing newline. No input sent. Type the URL alone, then use a separate Enter key action if navigation is intended.')

        # Calculator is a keypad, not a text editor. Clipboard paste can be
        # ignored or parse only part of an expression without raising an error.
        names = {el.name for el in screen_state.elements if el.is_enabled}
        calculator = ('Calculator' in screen_state.foreground_window_title and
                      {'Equals', 'Plus', 'Multiply by'}.issubset(names))
        if calculator:
            if action.target_tag is not None or not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', text):
                return ExecutionResult(False, 'Calculator type accepts one number without a target. Use observed operator buttons separately; inspect the display after each action.')
            pyautogui.write(text, interval=0.05)
            return ExecutionResult(True, f'Number keys sent: {text!r}; display must be verified.')

        # Click the target element first if specified, to grab focus
        if action.target_tag is not None:
            el = self._find_element(action.target_tag, screen_state)
            if el is not None and el.is_enabled and (el.role in ('EditControl', 'ComboBoxControl') or el.is_editable):
                if el.focus_only:
                    if not self._focus_accessible_editor(el, screen_state):
                        return ExecutionResult(False, 'The accessibility editor could not be focused and verified; no text input sent.')
                else:
                    pyautogui.click(*el.center)
                time.sleep(0.15)   # Let the click settle before typing
                if el.runtime_id and not self._typing_focus_matches(el):
                    return ExecutionResult(False, f'Typing focus did not match [{el.tag}] {el.role} {el.name!r}; no text or select-all input sent. Reobserve the field.')
                # Token/autocomplete controls may handle Ctrl+A by moving
                # focus to their token container. Do not select an unknown
                # value (or an empty one) as if it were a plain text editor.
                # A known text value still supports explicit replacement.
                if el.role != 'ComboBoxControl' or el.text:
                    pyautogui.hotkey('ctrl', 'a')
                    time.sleep(0.05)
            else:
                return ExecutionResult(False, 'Typing target is missing, disabled or not an editable control. Refresh the observation and choose an editable target.')

        # Clipboard paste is far more reliable than write() for unicode and is
        # much faster for long strings. But the clipboard belongs to the user —
        # snapshot it and put it back so an automated run doesn't silently
        # destroy whatever they had copied.
        readback = None
        verify_field = action.target_tag is not None and bool(el.runtime_id)
        direct_keys = (action.target_tag is not None and el.role == 'ComboBoxControl'
                       and len(text) <= 320 and all(32 <= ord(ch) <= 126 for ch in text))
        strategy = 'keyboard entry' if direct_keys else 'clipboard paste'
        previous = None
        clipboard = None
        input_started = False
        try:
            # Selection and clipboard preparation can yield to another window.
            # Check again at the input boundary, not only before Ctrl+A.
            def input_target_current():
                if screen_state.foreground_window_handle:
                    actual = self._windows().foreground()
                    if not actual or actual.handle != screen_state.foreground_window_handle:
                        return False
                return not verify_field or self._typing_focus_matches(el)

            if direct_keys:
                if not input_target_current():
                    return ExecutionResult(False, 'Typing focus changed before keyboard entry; no text sent. Reobserve the target.')
                input_started = True
                pyautogui.write(text, interval=.03)
            else:
                import pyperclip as clipboard
                previous = clipboard.paste()
                clipboard.copy(text)
                if not input_target_current():
                    return ExecutionResult(False, 'Typing focus changed before paste; no text sent. Reobserve the target.')
                input_started = True
                pyautogui.hotkey('ctrl', 'v')
            if verify_field:
                for _ in range(5):
                    time.sleep(.1)
                    readback = self._focused_text(el)
                    if readback is not None and readback.replace('\r\n', '\n') == text.replace('\r\n', '\n'):
                        break
            else:
                time.sleep(.05)
        except Exception as exc:
            return ExecutionResult(False, f'{strategy} failed ({type(exc).__name__}); no alternate input method was replayed.',
                                   input_unverified=input_started)
        finally:
            if clipboard is not None and previous is not None:
                try:
                    clipboard.copy(previous)
                except Exception:
                    pass
        target_description = (f'[{el.tag}] {el.role} {el.name!r}'
                              if action.target_tag is not None else 'current keyboard focus (no explicit field selected)')
        target_description += f' via {strategy}'
        if verify_field:
            if readback is None:
                focus_detail = ('Target focus still matches; accessibility value is unavailable.'
                                if input_target_current() else
                                'Target focus no longer matches after input; delivery is unverified.')
                return ExecutionResult(False, f'Text input attempted in {target_description}, but field readback is unavailable. {focus_detail} Do not repeat input blindly; inspect this field before proceeding.', input_unverified=True)
            if readback.replace('\r\n', '\n') != text.replace('\r\n', '\n'):
                return ExecutionResult(False, f'Text input readback mismatch in {target_description}. Observed value: {readback[:300]!r}. Inspect this field before proceeding; no automatic retry was sent.', input_unverified=True)
            return ExecutionResult(True, f'Text readback matched in {target_description}: {text!r}. Autocomplete acceptance and saving are not yet verified.')
        return ExecutionResult(True, f"Text input sent to {target_description}: {text!r}; visible content must be verified.")

    @staticmethod
    def _focused_text(element):
        """Read only the verified focused editor; None means unavailable."""
        from .perception import auto
        try:
            if not DesktopController._typing_focus_matches(element):
                return None
            node = auto.GetFocusedControl()
            # ValuePattern is the primary source for input/combobox values.
            try:
                pattern = node.GetValuePattern()
                if pattern is not None and isinstance(pattern.Value, str):
                    return pattern.Value
            except Exception:
                pass
            pattern = node.GetTextPattern()
            if pattern is not None:
                value = pattern.DocumentRange.GetText(50001)
                return value if isinstance(value, str) else None
        except Exception:
            pass
        return None

    @staticmethod
    def _typing_focus_matches(element):
        """Accept the observed editor or its editable descendant, not a nearby field."""
        from .perception import auto
        try:
            focused = auto.GetFocusedControl()
            if (not focused.IsEnabled or focused.IsPassword or focused.IsOffscreen or
                    focused.ControlTypeName not in {'EditControl', 'ComboBoxControl', 'DocumentControl'}):
                return False
            if focused.ControlTypeName == 'DocumentControl':
                from .perception import VisualPerceptionEngine
                if not VisualPerceptionEngine._is_editable_document(focused):
                    return False
            seen = set()
            for _ in range(8):
                identity = tuple(focused.GetRuntimeId())
                if identity == tuple(element.runtime_id):
                    return True
                if not identity or identity in seen or focused.ControlTypeName == 'WindowControl':
                    break
                seen.add(identity)
                focused = focused.GetParentControl()
        except Exception:
            return False
        return False

    @staticmethod
    def _focus_accessible_editor(element, state):
        """Resolve the observed UIA identity in the same window; never guess a point."""
        from .perception import auto
        from collections import deque
        if not element.runtime_id or not state.foreground_window_handle or auto is None:
            return False
        try:
            root = auto.ControlFromHandle(state.foreground_window_handle)
            pending = deque([(root, 0)])
            visited = 0
            while pending and visited < 512:
                node, depth = pending.popleft()
                visited += 1
                if tuple(node.GetRuntimeId()) == tuple(element.runtime_id):
                    if node.ControlTypeName != 'EditControl' or not node.IsEnabled or node.IsPassword or node.IsOffscreen:
                        return False
                    node.SetFocus()
                    return tuple(auto.GetFocusedControl().GetRuntimeId()) == tuple(element.runtime_id)
                if depth < 40:
                    for child in node.GetChildren():
                        if len(pending) + visited >= 512:
                            break
                        pending.append((child, depth + 1))
        except Exception:
            return False
        return False

    def _drag_and_drop(self, action: NextAction, screen_state: ScreenState) -> ExecutionResult:
        """SDS M5 FE-1: drag-and-drop between two accessibility-tree elements.

        `target_tag` is the drag source; `text` holds the destination tag number.
        """
        if action.target_tag is None:
            return ExecutionResult(False, "drag requires target_tag (the source element)")
        try:
            dest_tag = int(str(action.text).strip())
        except (TypeError, ValueError):
            return ExecutionResult(False, f"drag requires a destination tag in text, got {action.text!r}")

        source = self._find_element(action.target_tag, screen_state)
        dest = self._find_element(dest_tag, screen_state)
        if source is None:
            return ExecutionResult(False, f"Drag source tag {action.target_tag} not on screen")
        if dest is None:
            return ExecutionResult(False, f"Drag destination tag {dest_tag} not on screen")
        if source.focus_only or dest.focus_only:
            return ExecutionResult(False, 'A focus-only editor has no rectangle for a drag action.')

        sx, sy = source.center
        dx, dy = dest.center
        pyautogui.moveTo(sx, sy)
        pyautogui.mouseDown()
        # Move in steps — instant jumps are ignored by many drop targets.
        pyautogui.moveTo(dx, dy, duration=0.4)
        pyautogui.mouseUp()
        return ExecutionResult(
            True, f"Dragged [{source.tag}] '{source.name}' onto [{dest.tag}] '{dest.name}'"
        )

    def _key(self, combo: str) -> ExecutionResult:
        if not combo:
            return ExecutionResult(False, "key action requires text")
        keys = [k.strip().lower() for k in combo.split("+")]
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)
        # Give the OS time to handle the keystroke and open any dialog/window
        # before the next perception step runs its accessibility tree walk.
        time.sleep(0.6)
        return ExecutionResult(True, f"Pressed key: {combo}")

    def _scroll(self, action: NextAction, screen_state: ScreenState) -> ExecutionResult:
        try:
            amount = int(action.text) if action.text else -3
        except ValueError:
            amount = -3
        if action.target_tag is not None:
            el = self._find_element(action.target_tag, screen_state)
            if el:
                pyautogui.moveTo(*el.center)
        pyautogui.scroll(amount)
        return ExecutionResult(True, f"Scrolled {amount}")
