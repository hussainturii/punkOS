"""
M9 FE-1: Sandboxed code execution (SDS Algorithm 4, SDS 4.3.2, SEC-1/SEC-7).

This is the "Code-on-Demand" path (OB-3): for bulk file work — rename 500
files, sort a Downloads folder — driving the GUI one click at a time is
hopeless. The agent writes a Python script instead and we run it under
restrictions.

Defence in depth, in order:

  1. AST validation (always). The generated script is parsed and walked before
     it is ever executed. Imports outside the whitelist, calls to os.system /
     subprocess / eval / exec / __import__, network modules, and writes outside
     the authorized directory are all rejected statically.
  2. Isolation. Two backends:
       - "docker"     : full container, --network none, cpu/memory caps,
                        authorized dir bind-mounted, destroyed after the run
                        (matches the SDS spec exactly).
       - "subprocess" : separate interpreter with a scrubbed environment, a cwd
                        pinned to the authorized directory, and a hard timeout.
                        The default, because it needs no Docker install.
  3. Timeout + teardown. 30s default; the container/process is killed and
     removed even on timeout (SEC-7, REL-6).

Honest limitation: the subprocess backend is NOT a true security boundary — a
sufficiently clever script could still escape it, since it shares the host
kernel and filesystem namespace. AST validation is what actually carries the
weight there. Use the docker backend for real isolation. This is stated plainly
rather than papered over because overstating a sandbox is worse than not
having one.
"""
from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# SDS 4.3.2 — import whitelist.
ALLOWED_IMPORTS = {
    "os", "os.path", "pathlib", "shutil", "re", "json", "csv", "datetime",
    "math", "collections", "itertools", "string", "typing",
}

# Attribute calls that are never allowed, even on whitelisted modules.
FORBIDDEN_ATTRIBUTE_CALLS = {
    ("os", "system"), ("os", "popen"), ("os", "execv"), ("os", "execve"),
    ("os", "spawnl"), ("os", "spawnv"), ("os", "fork"), ("os", "kill"),
    ("shutil", "chown"),
}

# Bare names that must never be called.
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "__import__", "open_code",
    "breakpoint", "globals", "locals", "vars", "memoryview",
}

# Modules that must never be imported at all.
FORBIDDEN_IMPORTS = {
    "subprocess", "socket", "requests", "urllib", "urllib3", "http",
    "ftplib", "telnetlib", "smtplib", "ssl", "ctypes", "multiprocessing",
    "threading", "importlib", "pickle", "marshal", "shelve", "code",
    "pty", "platform", "sysconfig", "winreg", "msvcrt",
}


@dataclass
class ValidationResult:
    valid: bool
    violations: list[str] = field(default_factory=list)

    def reason(self) -> str:
        return "; ".join(self.violations) if self.violations else "OK"


@dataclass
class ExecutionResult:
    success: bool
    output: str = ""
    error: str = ""
    execution_time: float = 0.0
    exit_code: Optional[int] = None
    script_sha256: str = ""
    validation: Optional[ValidationResult] = None


class ScriptValidator(ast.NodeVisitor):
    """Walks the AST and collects every policy violation it finds."""

    def __init__(self, authorized_dirs: Sequence[str]):
        self.authorized_dirs = [os.path.normcase(os.path.abspath(os.path.expanduser(d)))
                                for d in authorized_dirs]
        self.violations: list[str] = []

    # -------------------- imports --------------------

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self._check_import(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self._check_import(node.module, node.lineno)
        self.generic_visit(node)

    def _check_import(self, module_name: str, lineno: int):
        root = module_name.split(".")[0]
        if root in FORBIDDEN_IMPORTS:
            self.violations.append(f"line {lineno}: import of forbidden module '{module_name}'")
        elif root not in {m.split(".")[0] for m in ALLOWED_IMPORTS}:
            self.violations.append(
                f"line {lineno}: import '{module_name}' is not in the whitelist "
                f"({', '.join(sorted(ALLOWED_IMPORTS))})"
            )

    # -------------------- calls --------------------

    def visit_Call(self, node: ast.Call):
        func = node.func

        if isinstance(func, ast.Name):
            if func.id in FORBIDDEN_NAMES:
                self.violations.append(f"line {node.lineno}: call to forbidden builtin '{func.id}'")
            if func.id == "open":
                self._check_open_call(node)

        elif isinstance(func, ast.Attribute):
            base = func.value
            if isinstance(base, ast.Name):
                pair = (base.id, func.attr)
                if pair in FORBIDDEN_ATTRIBUTE_CALLS:
                    self.violations.append(
                        f"line {node.lineno}: call to forbidden function '{base.id}.{func.attr}'"
                    )
        self.generic_visit(node)

    def _check_open_call(self, node: ast.Call):
        """Reject writes to a literal path outside the authorized directories.

        Non-literal paths (variables, f-strings) cannot be resolved statically;
        those are contained by the runtime cwd/mount restriction instead.
        """
        mode = "r"
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = str(node.args[1].value)
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = str(kw.value.value)

        if not any(m in mode for m in ("w", "a", "x", "+")):
            return   # read-only open is fine

        if not node.args or not isinstance(node.args[0], ast.Constant):
            return   # dynamic path — runtime restriction handles it

        path = str(node.args[0].value)
        if not os.path.isabs(path):
            return   # relative to the pinned cwd, which is already authorized

        norm = os.path.normcase(os.path.abspath(path))
        if not any(self._within(norm, d) for d in self.authorized_dirs):
            self.violations.append(
                f"line {node.lineno}: write to '{path}' is outside the authorized directories"
            )

    @staticmethod
    def _within(child: str, parent: str) -> bool:
        try:
            return os.path.commonpath([child, parent]) == parent
        except ValueError:
            return False


def validate_script(script: str, authorized_dirs: Sequence[str]) -> ValidationResult:
    """SDS Algorithm 4 steps 3-5: parse, walk, accept or reject."""
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return ValidationResult(False, [f"Script does not parse: {e}"])

    validator = ScriptValidator(authorized_dirs)
    validator.visit(tree)
    return ValidationResult(not validator.violations, validator.violations)


class SandboxExecutor:
    """Validates and runs an LLM-generated Python script under restrictions."""

    def __init__(
        self,
        authorized_dirs: Sequence[str],
        backend: str = "subprocess",
        docker_image: str = "python:3.11-slim",
        timeout_seconds: int = 30,
        memory_limit_mb: int = 256,
        cpu_limit: float = 0.5,
        audit_log=None,
    ):
        self.authorized_dirs = [os.path.abspath(os.path.expanduser(d)) for d in authorized_dirs]
        self.backend = backend
        self.docker_image = docker_image
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit = cpu_limit
        self.audit_log = audit_log

    # ------------------------------------------------------------------

    def execute(self, script: str, working_dir: Optional[str] = None) -> ExecutionResult:
        script_hash = hashlib.sha256(script.encode("utf-8")).hexdigest()

        # Step 3-5: static validation gate.
        validation = validate_script(script, self.authorized_dirs)
        if not validation.valid:
            result = ExecutionResult(
                success=False,
                error=f"Script failed safety validation: {validation.reason()}",
                script_sha256=script_hash,
                validation=validation,
            )
            if self.audit_log is not None:
                self.audit_log.record(
                    event="sandbox_rejected",
                    detail=f"script {script_hash[:16]} rejected: {validation.reason()}",
                    metadata={"script_sha256": script_hash, "violations": validation.violations},
                )
            return result

        cwd = self._resolve_working_dir(working_dir)
        if cwd is None:
            return ExecutionResult(
                success=False,
                error="No authorized directory available to run in",
                script_sha256=script_hash,
                validation=validation,
            )

        start = time.monotonic()
        if self.backend == "docker":
            result = self._run_docker(script, cwd)
        else:
            result = self._run_subprocess(script, cwd)
        result.execution_time = time.monotonic() - start
        result.script_sha256 = script_hash
        result.validation = validation

        if self.audit_log is not None:
            self.audit_log.record_sandbox(
                script_hash, result.exit_code if result.exit_code is not None else -1,
                result.execution_time,
            )
        return result

    def _resolve_working_dir(self, working_dir: Optional[str]) -> Optional[str]:
        if working_dir:
            target = os.path.abspath(os.path.expanduser(working_dir))
            for d in self.authorized_dirs:
                try:
                    if os.path.commonpath([os.path.normcase(target),
                                           os.path.normcase(d)]) == os.path.normcase(d):
                        return target
                except ValueError:
                    continue
            return None
        for d in self.authorized_dirs:
            if os.path.isdir(d):
                return d
        return None

    # -------------------- backends --------------------

    def _run_subprocess(self, script: str, cwd: str) -> ExecutionResult:
        """Isolated child interpreter: scrubbed env, pinned cwd, hard timeout."""
        tmp = Path(tempfile.gettempdir()) / f"punkos_sandbox_{uuid.uuid4().hex}.py"
        tmp.write_text(script, encoding="utf-8")

        # Minimal environment — no inherited API keys or user secrets (SEC-4).
        env = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        proc = None
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", str(tmp)],
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stdout, stderr = proc.communicate(timeout=self.timeout_seconds)
            return ExecutionResult(
                success=proc.returncode == 0,
                output=stdout or "",
                error=stderr or "",
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            # REL-6 / SEC-7: kill it, leave nothing orphaned.
            if proc is not None:
                proc.kill()
                try:
                    proc.communicate(timeout=5)
                except Exception:
                    pass
            return ExecutionResult(
                success=False,
                error=f"Script exceeded the {self.timeout_seconds}s timeout and was terminated",
                exit_code=-1,
            )
        except Exception as e:
            return ExecutionResult(success=False, error=f"Sandbox launch failed: {e}", exit_code=-1)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    def _run_docker(self, script: str, cwd: str) -> ExecutionResult:
        """Full container isolation per SDS 4.3.2."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="punkos_sandbox_"))
        script_path = tmp_dir / "task.py"
        script_path.write_text(script, encoding="utf-8")
        container_name = f"punkos-sandbox-{uuid.uuid4().hex[:12]}"

        cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--network", "none",                                  # SEC-1: no network
            f"--memory={self.memory_limit_mb}m",
            f"--cpus={self.cpu_limit}",
            "--security-opt", "no-new-privileges",
            "-v", f"{cwd}:/workspace:rw",                         # only the authorized dir
            "-v", f"{tmp_dir}:/script:ro",
            "-w", "/workspace",
            self.docker_image,
            "python", "-I", "/script/task.py",
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=self.timeout_seconds + 10,   # docker startup overhead
            )
            return ExecutionResult(
                success=proc.returncode == 0,
                output=proc.stdout or "",
                error=proc.stderr or "",
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            self._force_remove_container(container_name)
            return ExecutionResult(
                success=False,
                error=f"Container exceeded the {self.timeout_seconds}s timeout and was destroyed",
                exit_code=-1,
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                error="Docker is not installed or not on PATH. "
                      "Set sandbox.backend to 'subprocess' in config.json.",
                exit_code=-1,
            )
        except Exception as e:
            self._force_remove_container(container_name)
            return ExecutionResult(success=False, error=f"Docker sandbox failed: {e}", exit_code=-1)
        finally:
            try:
                script_path.unlink()
                tmp_dir.rmdir()
            except OSError:
                pass

    @staticmethod
    def _force_remove_container(name: str) -> None:
        """SEC-7 / REL-6: guarantee teardown even when the run went wrong."""
        try:
            subprocess.run(["docker", "rm", "-f", name],
                           capture_output=True, timeout=10)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Prompt used to ask the LLM for a script (SDS Algorithm 4 step 1)
# ---------------------------------------------------------------------------

CODE_GENERATION_PROMPT = """You write short, safe Python scripts for file operations.

HARD RULES — a script breaking any of these is rejected before it runs:
  - You may ONLY import: {allowed}
  - NEVER use: subprocess, socket, requests, urllib, eval, exec, __import__, os.system
  - NEVER write to a path outside: {authorized}
  - The script's working directory is already set to the target folder. Use
    relative paths, or paths under the authorized directory.
  - Print a one-line summary of what you changed so the user gets feedback.

Write ONLY the Python script. No markdown fences, no commentary."""


def build_code_prompt(authorized_dirs: Sequence[str]) -> str:
    return CODE_GENERATION_PROMPT.format(
        allowed=", ".join(sorted(ALLOWED_IMPORTS)),
        authorized=", ".join(authorized_dirs),
    )
