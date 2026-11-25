"""Deterministic launch/focus of an explicitly named supported desktop app.

Only fixed Windows app executables; no shell, shortcuts or model-generated commands.
Native calls are isolated behind a backend so policy is testable without input.
"""
from dataclasses import dataclass
import os
import re
import time

from .interaction_policy import InteractionPolicy


@dataclass(frozen=True)
class Window:
    handle: int
    title: str
    executable: str


@dataclass(frozen=True)
class HandoffResult:
    success: bool
    message: str
    complete: bool = False
    target_app: str = ''
    target_verified: bool = False


APPS = {
    'calculator': {'calculator.exe', 'calculatorapp.exe'},
    'notepad': {'notepad.exe'},
    'vscode': {'code.exe'},
    'chrome': {'chrome.exe'},
    'excel': {'excel.exe'},
}

ALIASES = {'vscode': r'vs\s*code|visual studio code', 'chrome': r'chrome|google chrome',
           'excel': r'excel|microsoft excel', 'notepad': 'notepad', 'calculator': 'calculator'}


def requested_apps(goal):
    return [name for name, pattern in ALIASES.items() if re.search(r'\b(?:' + pattern + r')\b', goal, re.I)]


def hosted_executable(executable, child_executables):
    """Resolve UWP frames only from a unique supported child executable."""
    if executable.casefold() != 'applicationframehost.exe':
        return executable
    supported = set().union(*APPS.values())
    matches = {name.casefold() for name in child_executables if name.casefold() in supported}
    return next(iter(matches)) if len(matches) == 1 else executable

LAUNCH_PREFIX = r'^(?:(?:please|can you|could you)\s+)?(?:open|launch|start)\s+(?:the\s+)?(?:windows\s+)?'


def launch_requested(goal, app):
    return bool(re.match(LAUNCH_PREFIX + '(?:' + ALIASES.get(app, re.escape(app)) + r')\b', goal.strip(), re.I))


def launch_only(goal, app):
    return bool(re.fullmatch(LAUNCH_PREFIX + '(?:' + ALIASES.get(app, re.escape(app)) + r')(?:\s+app)?[.!?\s]*', goal.strip(), re.I))


class DesktopHandoff:
    def __init__(self, backend=None):
        self.backend = backend

    def reuse_browser(self, goal, match_open_title=False):
        """Focus an existing browser, or return None when none matches.

        Never launches, attaches a debugger, or copies browser profile data.
        An ambiguous/inaccessible existing session is not permission to replace it.
        """
        reference = (re.search(r'\b(?:currently\s+open|already\s+open|open)\s+([^.!?\n]{1,100})', goal, re.I)
                     if match_open_title else None)
        if match_open_title and not reference:
            return None
        try:
            backend = self.backend or WindowsBackend()
            browsers = {'chrome': 'chrome.exe', 'edge': 'msedge.exe', 'firefox': 'firefox.exe'}
            # Ignore URLs/email addresses when selecting a named application.
            prose = re.sub(r'https?://\S+|\S+@\S+', '', goal, flags=re.I)
            named = [name for name in browsers if re.search(r'\b'+name+r'\b', prose, re.I)]
            if len(named) > 1:
                return HandoffResult(False, 'Multiple browser applications requested; select one browser for this task.')
            executables = {browsers[n] for n in named} if named else set(browsers.values())
            windows = [w for w in backend.windows() if w.executable.casefold() in executables]
            if match_open_title:
                # Resolve a page named by the user against observed titles,
                # never a fixed service-to-browser mapping.
                words = re.findall(r'\b[\w-]+\b', reference.group(1).casefold())[:6]
                generic = {'the', 'my', 'a', 'an', 'window', 'browser', 'page', 'tab',
                           'composer', 'form', 'and', 'in', 'on', 'please'}
                terms = {w for w in words if len(w) >= 3 and w not in generic}
                windows = [w for w in windows if terms & set(re.findall(r'\b[\w-]+\b', w.title.casefold()))]
            if not windows:
                return None
            foreground = backend.foreground()
            active = next((w for w in windows if foreground and w.handle == foreground.handle), None)
            if active is None and len(windows) > 1:
                return HandoffResult(False, 'Multiple browser windows are open. Bring the intended browser window to the foreground and retry; no guest session was launched.')
            target = active or windows[0]
            if active is None:
                backend.activate(target.handle)
            actual = backend.foreground()
            for _ in range(20):
                if actual and actual.handle == target.handle and actual.executable.casefold() == target.executable.casefold():
                    return HandoffResult(True, 'Using the existing browser window through desktop controls; its session is retained.', target_verified=True)
                time.sleep(.1)
                actual = backend.foreground()
            return HandoffResult(False, 'Could not focus the existing browser window. No guest session was launched.')
        except Exception as exc:
            return HandoffResult(False, f'Existing browser detection failed ({type(exc).__name__}); no guest session was launched.')

    def __call__(self, goal):
        try:
            backend = self.backend or WindowsBackend()
            foreground = backend.foreground()
            requested = requested_apps(goal)
            default_editor = not requested and bool(re.search(r'\b(?:create|write|save)\b', goal, re.I) and
                               re.search(r'\b[\w-]+\.(?:py|txt|md|json|csv|js|html|css)\b', goal, re.I))
            if default_editor:
                requested = ['notepad']
            mouse_launch = InteractionPolicy(goal).scope in {'all', 'launch'}
            if not requested:
                if re.match(LAUNCH_PREFIX, goal.strip(), re.I):
                    if backend.prepare_launcher():
                        return HandoffResult(True, 'Observed Start menu opened by mouse. Locate the requested app using current controls.')
                    return HandoffResult(False, 'Could not expose an observed Start menu. No guessed launcher input was sent.')
                if foreground and not foreground.title.casefold().startswith('punkos'):
                    return HandoffResult(True, 'Using the current foreground application.')
                return HandoffResult(False, 'Name the target application or a text file to create; the current foreground window is PunkOS.')
            if len(requested) != 1:
                return HandoffResult(False, 'This handoff supports one target app per task. Split the request into separate app tasks.')
            app = requested[0]
            windows = [w for w in backend.windows() if w.executable.casefold() in APPS[app]]
            if not windows:
                if not (launch_requested(goal, app) or default_editor):
                    return HandoffResult(False, f'No open {app.title()} window was found. Ask to open {app.title()} first.')
                if mouse_launch:
                    if backend.prepare_launcher():
                        return HandoffResult(True, f'Start menu opened by mouse. Locate {app} using observed controls.', target_app=app)
                    return HandoffResult(False, 'Mouse launch could not expose the Start menu; no executable launch was used.')
                try:
                    backend.launch(app)
                except FileNotFoundError:
                    if backend.prepare_launcher():
                        return HandoffResult(True, f'Use the observed Start menu to locate {app}.', target_app=app)
                    raise
                for _ in range(40):
                    windows = [w for w in backend.windows() if w.executable.casefold() in APPS[app]]
                    if windows:
                        break
                    time.sleep(0.2)
                if not windows:
                    return HandoffResult(False, f'{app.title()} launch was requested, but no supported window appeared within 8 seconds.')
            if len(windows) != 1:
                return HandoffResult(False, f'Multiple {app.title()} windows are open. Keep one target window open and retry.')
            target = windows[0]
            if not foreground or foreground.handle != target.handle:
                if mouse_launch:
                    if not backend.click_taskbar(app):
                        detail = getattr(backend, 'mouse_handoff_detail', '')
                        return HandoffResult(False, f'Mouse-only handoff: no unique visible {app.title()} taskbar button was available. Keep one target window open and its taskbar button visible, then retry. {detail}')
                else:
                    backend.activate(target.handle)
            actual = backend.foreground()
            # Store apps can finish activating after SetForegroundWindow returns.
            # Wait for this exact window, never accept another app by its title.
            for _ in range(20):
                if actual and actual.handle == target.handle:
                    break
                time.sleep(0.1)
                actual = backend.foreground()
            if not actual or actual.handle != target.handle:
                return HandoffResult(False, f'Windows did not focus the expected {app.title()} window after handoff. No model call or task input was issued.')
            return HandoffResult(True, f'Verified foreground app: {app.title()}.', launch_only(goal, app), app, True)
        except Exception as exc:
            return HandoffResult(False, f'Window handoff unavailable ({type(exc).__name__}). No model call or further task input was issued.')


class WindowsBackend:
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('Windows required')
        import ctypes as c
        from ctypes import wintypes as w
        self.c, self.w = c, w
        self.user = c.WinDLL('user32', use_last_error=True)
        self.kernel = c.WinDLL('kernel32', use_last_error=True)
        self.callback_type = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        signatures = [
            (self.user.EnumWindows, [self.callback_type, w.LPARAM], w.BOOL),
            (self.user.EnumChildWindows, [w.HWND, self.callback_type, w.LPARAM], w.BOOL),
            (self.user.IsWindowVisible, [w.HWND], w.BOOL),
            (self.user.GetWindow, [w.HWND, w.UINT], w.HWND),
            (self.user.GetWindowTextW, [w.HWND, w.LPWSTR, c.c_int], c.c_int),
            (self.user.GetWindowThreadProcessId, [w.HWND, c.POINTER(w.DWORD)], w.DWORD),
            (self.user.GetForegroundWindow, [], w.HWND),
            (self.user.IsIconic, [w.HWND], w.BOOL),
            (self.user.ShowWindow, [w.HWND, c.c_int], w.BOOL),
            (self.user.SetForegroundWindow, [w.HWND], w.BOOL),
            (self.kernel.OpenProcess, [w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            (self.kernel.QueryFullProcessImageNameW, [w.HANDLE, w.DWORD, w.LPWSTR, c.POINTER(w.DWORD)], w.BOOL),
            (self.kernel.CloseHandle, [w.HANDLE], w.BOOL),
            (self.kernel.GetWindowsDirectoryW, [w.LPWSTR, w.UINT], w.UINT),
        ]
        for fn, args, result in signatures:
            fn.argtypes, fn.restype = args, result

    def _window(self, handle, resolve_host=True):
        if not handle:
            return None
        title = self.c.create_unicode_buffer(1024)
        self.user.GetWindowTextW(handle, title, len(title))
        pid = self.w.DWORD()
        self.user.GetWindowThreadProcessId(handle, self.c.byref(pid))
        process = self.kernel.OpenProcess(0x1000, False, pid.value)
        executable = ''
        if process:
            try:
                path = self.c.create_unicode_buffer(32768)
                size = self.w.DWORD(len(path))
                if self.kernel.QueryFullProcessImageNameW(process, 0, path, self.c.byref(size)):
                    executable = path.value.rsplit('\\', 1)[-1]
            finally:
                self.kernel.CloseHandle(process)
        if executable.casefold() == 'applicationframehost.exe':
            children = []
            @self.callback_type
            def collect_child(child, _):
                item = self._window(child, resolve_host=False)
                if item:
                    children.append(item.executable)
                return True
            if resolve_host:
                self.user.EnumChildWindows(handle, collect_child, 0)
                executable = hosted_executable(executable, children)
        return Window(int(handle), title.value, executable)

    def windows(self):
        result = []
        @self.callback_type
        def collect(handle, _):
            # Owned dialogs/tooltips belong to a document window; counting them
            # as independent app windows creates false ambiguity. Activation is
            # still verified against the exact chosen handle before any input.
            if self.user.IsWindowVisible(handle) and not self.user.GetWindow(handle, 4):  # GW_OWNER
                item = self._window(handle)
                if item and item.title:
                    result.append(item)
            return True
        if not self.user.EnumWindows(collect, 0):
            raise OSError('Could not enumerate windows')
        return result

    def foreground(self):
        return self._window(self.user.GetForegroundWindow())

    def click_taskbar(self, app):
        """One real mouse click on an observed shell button; never native focus.

        Only known app labels under Explorer taskbars are eligible. Re-read the
        button and hit-test immediately before input to reject stale/covered UI.
        The caller verifies the resulting foreground window independently.
        """
        from .taskbar_handoff import click_app_button
        from .mouse_window_focus import click_titlebar, minimize_own_dashboard
        self.mouse_handoff_detail = ''
        targets = [window for window in self.windows() if window.executable.casefold() in APPS[app]]
        if len(targets) == 1 and click_titlebar(targets[0].handle):
            return True
        if click_app_button(app, diagnostic=lambda message: setattr(self, 'mouse_handoff_detail', message)):
            return True
        if len(targets) == 1 and minimize_own_dashboard():
            time.sleep(0.2)
            foreground = self.foreground()
            if foreground and foreground.handle == targets[0].handle:
                return True
            if click_titlebar(targets[0].handle):
                return True
            self.mouse_handoff_detail += ' Dashboard minimized, but the target is still covered.'
        return False

    def launch(self, app):
        import subprocess
        from pathlib import Path
        if app not in {'notepad', 'calculator'}:
            relative = {'vscode': ['Microsoft VS Code/Code.exe'],
                        'chrome': ['Google/Chrome/Application/chrome.exe'],
                        'excel': ['Microsoft Office/root/Office16/EXCEL.EXE', 'Microsoft Office/Office16/EXCEL.EXE']}[app]
            roots = [Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs',
                     Path(os.environ.get('LOCALAPPDATA', '')), Path(os.environ.get('ProgramFiles', '')),
                     Path(os.environ.get('ProgramFiles(x86)', ''))]
            candidates = [root / name for root in roots if root.is_absolute() for name in relative]
            executable = next((p for p in candidates if p.is_file()), None)
            if executable is None:
                raise FileNotFoundError(f'{app} was not found in standard install locations')
            args = [str(executable)]
            if app == 'vscode':
                args.append('--force-renderer-accessibility')
            subprocess.Popen(args, shell=False)
            return
        executable = {'notepad': 'notepad.exe', 'calculator': 'calc.exe'}[app]
        directory = self.c.create_unicode_buffer(32768)
        length = self.kernel.GetWindowsDirectoryW(directory, len(directory))
        if not length or length >= len(directory):
            raise OSError('Windows directory unavailable')
        subprocess.Popen([os.path.join(directory.value, 'System32', executable)], shell=False)

    def prepare_launcher(self):
        from .taskbar_handoff import click_app_button
        from .mouse_window_focus import minimize_own_dashboard
        if not click_app_button('start'):
            if not minimize_own_dashboard() or not click_app_button('start'):
                return False
        for _ in range(15):
            foreground = self.foreground()
            if foreground and foreground.executable.casefold() == 'startmenuexperiencehost.exe':
                return True
            time.sleep(.1)
        return False

    def target_is_foreground(self, app):
        foreground = self.foreground()
        return bool(foreground and foreground.executable.casefold() in APPS.get(app, set()))

    def activate(self, handle):
        if self.user.IsIconic(handle):
            self.user.ShowWindow(handle, 9)  # SW_RESTORE
        self.user.SetForegroundWindow(handle)
        for _ in range(10):
            if self.user.GetForegroundWindow() == handle:
                return
            time.sleep(0.05)
