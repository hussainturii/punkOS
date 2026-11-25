"""Resolve explicit Desktop file destinations without creating or running files."""
import os
import re
from pathlib import Path


def desktop_path():
    if os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                               r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') as key:
                value, _ = winreg.QueryValueEx(key, 'Desktop')
            path = Path(os.path.expandvars(value))
            if path.is_absolute():
                return path
        except OSError:
            pass
    return Path.home() / 'Desktop'


_NAME = r'(?<![\w./\\:-])([\w-]+\.(?:py|txt|md|json|csv|js|html|css))(?!\w|\.\w)'


def _desktop_role(goal, verbs):
    candidates = []
    # Sentence punctuation is separate from the dots inside filenames.
    for clause in re.split(r'(?<=[.!?])\s+|\n+', goal):
        if not re.search(r'\bdesktop\b', clause, re.I):
            continue
        verb = re.search(r'\b(?:' + verbs + r')\b', clause, re.I)
        if verb:
            candidates.extend(re.findall(_NAME, clause[verb.end():], re.I))
    unique = list(dict.fromkeys(candidates))
    return str(desktop_path() / unique[0]) if len(unique) == 1 else None


def desktop_output(goal):
    # Associate filenames with writing clauses, never with an earlier open/read.
    return _desktop_role(goal, 'create|write|save')


def desktop_source(goal):
    return _desktop_role(goal, 'open|read')


def existing_file_stop(goal):
    """Resolve the explicit 'if it exists, stop' branch before any app input.

    None means continue. A (success, message) tuple is a terminal outcome.
    Unknown existence is never treated as permission to write.
    """
    instructions = re.sub(r'"[^"\n]*"|\x27[^\x27\n]*\x27', '', goal)
    condition = re.search(r'\bif\b[^!?\n]*?\bexists?\b[^.!?\n]*\bstop\b', instructions, re.I)
    if not condition or not re.search(r'\b(?:create|write|save)\b', goal, re.I):
        return None
    target = desktop_output(goal)
    if not target:
        paths = []
        for clause in re.split(r'(?<=[.!?])\s+|\n+', goal):
            verb = re.search(r'\b(?:create|write|save)\b', clause, re.I)
            if verb:
                paths.extend(re.findall(r'\b[a-z]:[\\/][^\r\n"<>|?*]+?\.(?:py|txt|md|json|csv|js|html|css)\b', clause[verb.end():], re.I))
        unique = list(dict.fromkeys(paths))
        target = unique[0] if len(unique) == 1 else None
    if not target:
        return False, 'Cannot resolve one exact destination for the requested stop-if-existing check. Specify a full file path; no task input was issued.'
    try:
        Path(target).lstat()
    except FileNotFoundError:
        return None
    except OSError:
        return False, f'Could not check whether the destination exists: {target}. Stopped rather than risk overwriting it.'
    return True, f'The destination already exists: {target}. Stopped as requested without writing to it.'
