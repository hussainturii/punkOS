"""Task-scoped interaction constraints, independent of the selected model."""
import re


def mouse_requested(goal):
    return bool(re.search(r'mouse[- ]only|(?:use|using|with|via)[^.\n]*?mouse|mouse interaction|interaction of mouse', goal, re.I))


class InteractionPolicy:
    def __init__(self, goal):
        self.goal = goal
        self.requested = mouse_requested(goal)
        self.launch_complete = False
        self.scope = 'none'
        if self.requested:
            self.scope = 'all'
            # An explicit whole-task restriction wins over local clauses.
            if not re.search(r'(?:entire|whole)\s+task|throughout|only\s+(?:the\s+)?mouse|no\s+keyboard', goal, re.I):
                clauses = re.split(r'\b(?:then|afterwards)\b|[.;\n]', goal, flags=re.I)
                mouse_clauses = [c for c in clauses if mouse_requested(c)]
                if mouse_clauses and all(re.search(r'\b(?:open|launch|start)\b', c, re.I) for c in mouse_clauses):
                    self.scope = 'launch'
                elif mouse_clauses and all(re.search(r'\bsave\b', c, re.I) for c in mouse_clauses):
                    self.scope = 'save'

    def restriction(self, action, state=None):
        if self.scope == 'all' and action.action_type in {'key', 'type'}:
            return 'The whole task is mouse-only; use observed mouse controls, not keyboard input.'
        if self.scope == 'launch' and not self.launch_complete and action.action_type in {'key', 'type'}:
            return 'Opening the app must use mouse interaction. Click observed launcher/app controls first.'
        saving_dialog = state is not None and 'save' in state.foreground_window_title.casefold()
        save_keys = {'ctrl+s', 'ctrl+shift+s', 'alt+f'}
        if saving_dialog:
            save_keys.update({'enter', 'return', 'alt+s'})
        if self.scope == 'save' and action.action_type == 'key' and action.text.lower() in save_keys:
            return 'The save step requires mouse interaction. Use the observed File/Save controls; filename text entry is allowed.'
        return ''

    def hint(self):
        if self.scope == 'none':
            return 'Use suitable keyboard shortcuts, typing, or mouse controls; no mouse restriction was requested.'
        if self.scope == 'launch' and self.launch_complete:
            return 'Mouse launch requirement is satisfied. Keyboard shortcuts and typing are allowed for remaining steps.'
        if self.scope == 'launch':
            return 'Open the requested app using observed mouse controls. After verified app opening, normal typing/shortcuts are allowed.'
        if self.scope == 'save':
            return 'Use mouse controls for saving; normal editing shortcuts and filename text entry are allowed.'
        return 'Use mouse controls throughout. Do not send keys or type; report a blocker if text entry cannot be achieved with observed controls.'
