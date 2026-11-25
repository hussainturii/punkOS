"""Single, explicit filesystem operations. Never execute model-generated code."""
import hashlib
import os
from pathlib import Path
import re

SCHEMA = {'type': 'object', 'properties': {
    'operation': {'type': 'string', 'enum': ['copy', 'move', 'rename', 'mkdir', 'unsupported']},
    'source': {'type': 'string'}, 'destination': {'type': 'string'},
    'reason': {'type': 'string'}}, 'required': ['operation', 'source', 'destination', 'reason']}
MAX_BYTES = 64 * 1024 * 1024


def explicit_paths(goal):
    return list(dict.fromkeys(value for value in re.findall(r'"([^"\n]+)"', goal)
                             if Path(value).is_absolute()))


def eligible(goal):
    instructions = re.sub(r'"[^"\n]*"', '', goal).lower()
    return bool(re.findall(r'"([^"\n]+)"', goal) and
                re.match(r'\s*(?:please\s+)?(?:copy|move|rename|create|make)\b', instructions) and
                not re.search(r'\b(?:browser|chrome|firefox|edge|explorer|mouse|keyboard|notepad|code|editor)\b', instructions) and
                (re.search(r'^\s*(?:please\s+)?(?:copy|move|rename)\b', instructions) or
                 re.search(r'\b(?:folder|directory)\b', instructions)))


def _path(value, safety):
    raw = Path(value)
    if not raw.is_absolute() or '..' in raw.parts or any(c in value for c in '*?\x00'):
        raise ValueError('Use an absolute path without traversal or wildcards.')
    if os.name == 'nt' and (':' in value[2:] or any(part.endswith((' ', '.')) for part in raw.parts[1:])):
        raise ValueError('Alternate streams and ambiguous Windows names are unsupported.')
    for part in (raw, *raw.parents):
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError('Linked files or directories are unsupported.')
    result = raw.resolve()
    allowed, reason = safety.is_path_allowed(str(result))
    if not allowed:
        raise PermissionError(reason)
    return result


def _digest(path):
    digest = hashlib.sha256()
    total = 0
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_BYTES:
                raise ValueError('File exceeds the 64 MiB verification limit.')
            digest.update(chunk)
    return digest.hexdigest()


def run_file_operation(goal, provider, safety, stopped=lambda: False, log=lambda _: None):
    history = []
    destination_created = None
    def finish(ok, message):
        return ok, message, history
    try:
        if stopped():
            return finish(False, 'File operation stopped before planning.')
        instructions = re.sub(r'"[^"\n]*"', '', goal)
        if re.search(r'\b(?:preview|dry.run|simulate|simulation)\b|\b(?:do not|don\x27t|never)\s+(?:copy|move|rename|create|make)\b', instructions, re.I):
            return finish(False, 'Preview or prohibited-operation request: nothing changed. This route executes only explicit operations.')
        paths = explicit_paths(goal)
        if not paths or len(paths) > 2:
            return finish(False, 'Specify one operation with quoted absolute paths (source and exact destination).')
        for value in paths:
            _path(value, safety)
        response = provider.call_with_schema(
            system_prompt='Plan ONE explicit filesystem operation: copy a regular file, move a regular file, '
                'rename a regular file, or create one empty folder. Use paths verbatim from the user. '
                'The destination must be an exact new path, not an existing directory. '
                'Use unsupported for multiple operations, conditional workflows other than stop-if-destination-exists, '
                'previews/dry runs, recursive work, deletion, overwriting, editing contents, or extra outcomes. '
                'For mkdir source must be empty. Do not ignore constraints or invent paths.',
            user_text=goal, json_schema=SCHEMA, schema_name='file_operation_plan',
            schema_description='Describe a single explicit operation without code.')
        if stopped():
            return finish(False, 'File operation stopped before changes.')
        if response.error:
            return finish(False, 'File operation planning failed: ' + response.error)
        spec = response.parsed_json or {}
        if not isinstance(spec, dict):
            raise ValueError('Invalid file operation plan.')
        kind, source, target = spec.get('operation'), spec.get('source'), spec.get('destination')
        if kind not in {'copy', 'move', 'rename', 'mkdir'}:
            return finish(False, 'Unsupported file operation; nothing changed. ' + str(spec.get('reason', '')))
        if target not in paths or (kind != 'mkdir' and source not in paths) or (kind == 'mkdir' and source != ''):
            raise ValueError('Plan contains a path not explicitly supplied by the user.')
        # A provider cannot change the requested action into a more destructive one.
        verb = re.match(r'\s*(?:please\s+)?(copy|move|rename|create|make)\b', goal, re.I)
        if not verb or kind != {'create': 'mkdir', 'make': 'mkdir'}.get(verb[1].lower(), verb[1].lower()):
            raise ValueError('Planned operation differs from the requested action.')
        dest = _path(target, safety)
        src = _path(source, safety) if kind != 'mkdir' else None
        if src == dest:
            raise ValueError('Source and destination are the same path.')
        if dest.exists():
            return finish(False, f'Destination already exists: {dest}. Nothing was overwritten.')
        if not dest.parent.is_dir():
            raise ValueError('Destination parent folder does not exist. Create it separately first.')
        if src is not None and (not src.is_file() or src.stat().st_size > MAX_BYTES):
            raise ValueError('Source must be a regular file of at most 64 MiB.')
        if kind == 'rename' and src.parent != dest.parent:
            raise ValueError('Rename must keep the same parent folder; request move to change folders.')
        log(f'File operation: {kind}; source: {src or "(none)"}; destination: {dest}')
        if stopped():
            return finish(False, 'File operation stopped before changes.')
        # Revalidate scope immediately before opening/mutating paths.
        _path(target, safety)
        if src is not None:
            _path(source, safety)
        if kind == 'mkdir':
            dest.mkdir(exist_ok=False)
            destination_created = dest
            if not dest.is_dir():
                raise OSError('Created directory could not be verified.')
        else:
            before = src.stat()
            digest = _digest(src)
            if kind in {'move', 'rename'}:
                # Windows rename fails if destination exists. Other platforms
                # have overwrite semantics, so do not use their rename here.
                if os.name != 'nt':
                    raise ValueError('Move/rename currently require Windows no-overwrite rename semantics.')
                os.rename(src, dest)
                destination_created = dest
                if src.exists() or _digest(dest) != digest:
                    raise OSError('Move verification failed; inspect the reported paths.')
            else:
                with src.open('rb') as reader, dest.open('xb') as writer:
                    destination_created = dest
                    total = 0
                    while chunk := reader.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_BYTES:
                            raise ValueError('Source grew beyond the 64 MiB limit; partial copy remains.')
                        if stopped():
                            raise InterruptedError('Copy stopped; a partial destination may remain.')
                        writer.write(chunk)
                    writer.flush()
                    os.fsync(writer.fileno())
                after = src.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or _digest(src) != digest or _digest(dest) != digest:
                    raise OSError('Source changed or copied content verification failed.')
        message = f'Verified {kind}: {src or "(new folder)"} -> {dest}.'
        if src is not None:
            message += f' SHA-256: {digest}.'
        history.append({'action': 'file_' + kind, 'success': True, 'outcome': message})
        return finish(True, message)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        message = f'File operation failed ({type(exc).__name__}): {exc}'
        if destination_created is not None:
            message += f' Destination may remain at {destination_created}; no automatic rollback was attempted.'
        history.append({'action': 'file_operation', 'success': False, 'outcome': message})
        return finish(False, message)
