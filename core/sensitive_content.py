"""Local recognition of credential-bearing screen text before model requests."""
import re


_TOKEN = re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AIza[A-Za-z0-9_-]{30,})')
_ASSIGNMENT = re.compile(
    r'''(?im)\b(?:[A-Z][A-Z0-9_]*_)?(?:API_KEY|ACCESS_TOKEN|SECRET_KEY|PASSWORD)\s*[=:]\s*["']?([^\s"']{8,})''')


def contains_credentials(text: str) -> bool:
    """Conservative recognizable patterns; not a general secret detector."""
    return bool(_TOKEN.search(text or '') or _ASSIGNMENT.search(text or ''))


def sensitive_screen(state) -> bool:
    return bool(getattr(state, 'sensitive_content_detected', False)) or contains_credentials(
        '\n'.join([state.foreground_window_title] +
                  [f'{e.name}\n{e.text}' for e in state.elements]))
