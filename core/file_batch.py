"""Bounded, reviewed file/folder plans; no generated code or implicit overwrites."""
import fnmatch
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from .file_operations import _path, _digest, MAX_BYTES, run_file_operation

PLAN = {'type': 'object', 'properties': {
    'operations': {'type': 'array', 'maxItems': 50, 'items': {'type': 'object', 'properties': {
        'operation': {'type': 'string', 'enum': ['copy', 'move', 'rename', 'mkdir']},
        'source': {'type': 'string'}, 'destination': {'type': 'string'},
        'pattern': {'type': 'string', 'description': 'Optional filename glob for immediate files in source folder; empty means whole source.'}
    }, 'required': ['operation', 'source', 'destination', 'pattern']}},
    'unsupported': {'type': 'string'}}, 'required': ['operations', 'unsupported']}


def quoted_paths(goal):
    return list(dict.fromkeys(re.findall(r'"([^"\n]+)"', goal)))


def resolve_reference(value, safety):
    from .file_destination import desktop_path
    path = Path(value).expanduser()
    if not path.is_absolute():
        parts = re.split(r'[\\/]', value)
        if parts[0].lower() in ('desktop', 'my desktop'):
            path = desktop_path().joinpath(*parts[1:])
        elif len(safety.authorized_dirs) == 1:
            path = Path(safety.authorized_dirs[0]) / path
        else:
            raise ValueError('Relative path is ambiguous with multiple authorized folders. Use a full path or Desktop/path.')
    return _path(str(path), safety)


def needs_batch(goal):
    values = quoted_paths(goal)
    prose = re.sub(r'"[^"\n]*"', '', goal)
    return (len(values) > 2 or any(not Path(p).is_absolute() for p in values) or
            bool(re.search(r'\b(?:desktop|folders?|directories|directory|batch|all|each|every|preview|dry.run)\b', prose, re.I)) or
            any(Path(p).is_dir() for p in values))


def tree_snapshot(path, safety, stopped):
    """Include empty directories as well as file hashes; reject links and excess."""
    records = {}
    pending = [path]
    total = 0
    while pending:
        if stopped():
            raise InterruptedError('Stopped while inspecting source tree.')
        current = pending.pop()
        _path(str(current), safety)
        relative = str(current.relative_to(path))
        if len(records) >= 256:
            raise ValueError('Plan exceeds 256 filesystem entries.')
        if current.is_dir():
            records[relative] = None
            with os.scandir(current) as entries:
                for entry in entries:
                    if len(pending) + len(records) >= 256:
                        raise ValueError('Source tree exceeds 256 entries.')
                    pending.append(Path(entry.path))
        elif current.is_file():
            total += current.stat().st_size
            if current.stat().st_size > MAX_BYTES or total > 256 * 1024 * 1024:
                raise ValueError('File/tree size exceeds 64 MiB per file or 256 MiB total.')
            records[relative] = _digest(current)
        else:
            raise ValueError(f'Unsupported or missing source: {current}')
    return records


def run_batch(goal, provider, safety, stopped=lambda: False, log=lambda _: None, confirm=None):
    history, completed = [], []
    active_destination = None
    try:
        if stopped():
            return False, 'File plan stopped.', history
        prose = re.sub(r'"[^"\n]*"', '', goal)
        preview = bool(re.search(r'\b(?:preview|dry.run|simulate)\b', prose, re.I))
        if not preview and re.search(r'\b(?:do not|never|don\x27t)\s+(?:copy|move|rename|create|make)\b', prose, re.I):
            raise ValueError('Request prohibits the operation; no changes made.')
        refs = {name: resolve_reference(name, safety) for name in quoted_paths(goal)}
        if re.search(r'\b(?:my\s+)?desktop\b', prose, re.I):
            refs['Desktop'] = resolve_reference('Desktop', safety)
        if not refs or len(refs) > 100:
            raise ValueError('Supply 1–100 quoted file/folder paths.')
        response = provider.call_with_schema(system_prompt=
            'Plan a bounded file operation from the user request. No deletion or overwrite is supported. '
            'Use only the supplied reference paths (their keys or resolved absolute values). '
            'For a filename batch use source folder, destination folder and a filename glob pattern. '
            'A pattern selects only immediate regular files, never subfolders. Empty pattern copies/moves '
            'the whole source, including a folder recursively. Whole-folder destination must be a new exact path. '
            'An existing destination directory is interpreted as a containing directory: the source basename is appended. '
            'For multiple individual files use one operation per explicit source/destination. '
            'mkdir creates an empty folder. Set unsupported and operations=[] for unsupported constraints. '
            'Do not silently omit part of the goal. Plans will be reviewed before execution.',
            user_text=json.dumps({'goal': goal, 'references': {k: str(v) for k, v in refs.items()}}),
            json_schema=PLAN, schema_name='file_batch_plan', schema_description='Reviewed file plan without code.')
        if stopped():
            raise InterruptedError('Stopped during planning; no files changed.')
        if response.error:
            raise ValueError('Planning failed: ' + response.error)
        spec = response.parsed_json
        if not isinstance(spec, dict) or not isinstance(spec.get('operations'), list) or spec.get('unsupported'):
            raise ValueError('Unsupported/invalid plan: ' + str(spec))
        if not 1 <= len(spec['operations']) <= 50:
            raise ValueError('Plan must contain 1–50 operations.')
        def reference(value):
            if value in refs:
                return refs[value]
            if value in {str(p) for p in refs.values()}:
                return _path(value, safety)
            raise ValueError('Model introduced a path not supplied by the user.')
        operations = []
        allowed_verbs = set(re.findall(r'\b(copy|move|rename|create|make)\b', prose.lower()))
        for item in spec['operations']:
            if not isinstance(item, dict) or any(not isinstance(item.get(k), str) for k in ('operation','source','destination','pattern')):
                raise ValueError('Malformed operation.')
            kind = item['operation']
            if kind not in {'copy','move','rename','mkdir'} or (kind not in allowed_verbs and not (kind == 'mkdir' and allowed_verbs & {'create','make'})):
                raise ValueError('Operation was not requested.')
            dest = reference(item['destination'])
            src = reference(item['source']) if kind != 'mkdir' else None
            pattern = item['pattern']
            if pattern:
                if kind not in {'copy', 'move'} or not src.is_dir() or any(c in pattern for c in '/\\:') or pattern in ('.','..'):
                    raise ValueError('Invalid filename-batch pattern.')
                if not dest.is_dir():
                    raise ValueError('Filename-batch destination must be an existing folder.')
                count = 0
                prior_count = len(operations)
                with os.scandir(src) as entries:
                    for entry in entries:
                        count += 1
                        if count > 1000:
                            raise ValueError('Source listing exceeds 1000 entries.')
                        if fnmatch.fnmatchcase(entry.name.casefold(), pattern.casefold()):
                            source = _path(str(src / entry.name), safety)
                            if source.is_file():
                                operations.append((kind, source, _path(str(dest / entry.name), safety)))
                if len(operations) == prior_count:
                    raise ValueError('No regular files match the requested pattern.')
            else:
                if src is not None and kind in {'copy','move'} and dest.is_dir():
                    dest = _path(str(dest / src.name), safety)
                operations.append((kind, src, dest))
        if not 1 <= len(operations) <= 100:
            raise ValueError('Expanded plan must contain 1–100 operations.')
        snapshots, targets = {}, set()
        for kind, src, dest in operations:
            if dest in targets or dest.exists():
                raise ValueError(f'Destination exists or repeats in plan: {dest}')
            if not dest.parent.is_dir():
                raise ValueError(f'Destination parent must already exist: {dest.parent}')
            if src is not None:
                if src == dest or dest.is_relative_to(src) or src.is_relative_to(dest):
                    raise ValueError('Source and destination overlap.')
                if kind == 'rename' and src.parent != dest.parent:
                    raise ValueError('Rename must keep the same parent.')
                snapshots[str(src)] = tree_snapshot(src, safety, stopped)
            targets.add(dest)
        sources = [src for _,src,_ in operations if src is not None]
        if len(set(sources)) != len(sources) or any(a != b and (a.is_relative_to(b) or b.is_relative_to(a)) for a in sources for b in sources):
            raise ValueError('Source operations overlap; split this into independent plans.')
        if any(dest == src or dest.is_relative_to(src) or src.is_relative_to(dest) for dest in targets for src in sources):
            raise ValueError('Plan writes into another source; split dependent operations into separate tasks.')
        entry_count = sum(len(v) for v in snapshots.values())
        if entry_count > 256:
            raise ValueError('Combined plan exceeds 256 entries.')
        if sum((Path(src) / relative).stat().st_size for src, records in snapshots.items()
               for relative, digest in records.items() if digest is not None) > 256 * 1024 * 1024:
            raise ValueError('Combined plan exceeds 256 MiB.')
        rows = [f'{kind}: {src or "(new folder)"} -> {dest}' for kind,src,dest in operations]
        plan_text = '\n'.join(rows) + f'\nSource entries: {entry_count}. No overwrites. Partial work remains on failure.'
        log(plan_text)
        history.append({'action': 'file_plan', 'success': True, 'outcome': plan_text})
        if preview:
            return True, 'Preview only; no files changed.\n' + plan_text, history
        if stopped() or confirm is None or not confirm(plan_text, operations):
            return False, 'File plan not approved or stopped. No changes made.', history
        # Recheck the entire plan after review, before its first mutation.
        for kind,src,dest in operations:
            _path(str(dest), safety)
            if dest.exists():
                raise ValueError(f'Destination appeared after review: {dest}')
            if src is not None and tree_snapshot(src, safety, stopped) != snapshots[str(src)]:
                raise ValueError(f'Source changed after review: {src}')
        for kind,src,dest in operations:
            if stopped():
                raise InterruptedError('Stopped; completed operations were not undone.')
            active_destination = str(dest)
            _path(str(dest), safety)
            if dest.exists():
                raise ValueError(f'Destination appeared during execution: {dest}')
            if src is not None and tree_snapshot(src, safety, stopped) != snapshots[str(src)]:
                raise ValueError(f'Source changed during execution: {src}')
            if src is not None and src.is_dir():
                before = snapshots[str(src)]
                if kind in {'move','rename'}:
                    if os.name != 'nt':
                        raise ValueError('Folder move/rename requires Windows no-overwrite rename.')
                    _path(str(src), safety)
                    _path(str(dest), safety)
                    os.rename(src, dest)
                else:
                    dest.mkdir(exist_ok=False)
                    for relative, digest in sorted(before.items(), key=lambda pair: (len(Path(pair[0]).parts), pair[0])):
                        if relative == '.':
                            continue
                        if stopped():
                            raise InterruptedError(f'Copy stopped; partial folder remains at {dest}')
                        out = _path(str(dest / relative), safety)
                        if digest is None:
                            out.mkdir(exist_ok=False)
                        else:
                            ok, message, trace = _execute_file('copy', src / relative, out, safety, stopped)
                            history.extend(trace)
                            if not ok:
                                raise OSError(message)
                    if tree_snapshot(src, safety, stopped) != before:
                        raise OSError('Source tree changed during copy.')
                if tree_snapshot(dest, safety, stopped) != before or (kind in {'move','rename'} and src.exists()):
                    raise OSError(f'Folder result could not be verified: {dest}')
            else:
                ok, message, trace = _execute_file(kind, src, dest, safety, stopped)
                history.extend(trace)
                if not ok:
                    raise OSError(message)
            completed.append(str(dest))
        return True, 'Verified file plan. Destinations:\n' + '\n'.join(completed), history
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        message = f'File plan failed: {exc}. Verified destinations: {completed}. Last attempted destination: {active_destination}. Partial destinations may remain; no automatic rollback.'
        history.append({'action': 'file_plan_failure', 'success': False, 'outcome': message})
        return False, message, history


def _execute_file(kind, source, dest, safety, stopped):
    spec = {'operation': kind, 'source': str(source) if source else '', 'destination': str(dest), 'reason': ''}
    class PreparedPlan:
        def call_with_schema(self, **kwargs):
            return SimpleNamespace(error=None, parsed_json=spec)
    goal = f'Create folder "{dest}"' if kind == 'mkdir' else f'{kind.title()} "{source}" to "{dest}"'
    return run_file_operation(goal, PreparedPlan(), safety, stopped)
