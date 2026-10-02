from __future__ import annotations

import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class JobSpec:
    name: str
    stage: str
    script: Path
    args: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    partition: str | None = None
    account: str | None = None
    time: str | None = None
    mem: str | None = None
    cpus_per_task: int | None = None
    output_path: str | None = None
    error_path: str | None = None
    gres: str | None = None
    requeue: bool = False
    # Slurm dependency type for ``dependencies``. Default is ``afterok`` (run
    # only on success of all parents); ``afterany`` lets a job act as a
    # post-mortem watchdog over its parents (runs regardless of outcome).
    dependency_type: str = "afterok"


class SlurmSubmitter:
    def __init__(self, *, max_attempts: int = 4, retry_delay_s: float = 2.0) -> None:
        self.max_attempts = max_attempts
        self.retry_delay_s = retry_delay_s

    def command_for(self, spec: JobSpec) -> list[str]:
        cmd = ["sbatch", "--parsable", f"--job-name={_safe_job_name(spec.name)}"]
        if spec.partition:
            cmd.append(f"--partition={spec.partition}")
        if spec.account:
            cmd.append(f"--account={spec.account}")
        if spec.time:
            cmd.append(f"--time={spec.time}")
        if spec.mem:
            cmd.append(f"--mem={spec.mem}")
        if spec.cpus_per_task:
            cmd.extend(["--cpus-per-task", str(spec.cpus_per_task)])
        if spec.gres:
            cmd.append(f"--gres={spec.gres}")
        if spec.requeue:
            cmd.append("--requeue")
        if spec.output_path:
            cmd.append(f"--output={spec.output_path}")
        if spec.error_path:
            cmd.append(f"--error={spec.error_path}")
        if spec.dependencies:
            dep_kind = spec.dependency_type or "afterok"
            cmd.append(f"--dependency={dep_kind}:{':'.join(spec.dependencies)}")
        cmd.append(str(spec.script))
        cmd.extend(spec.args)
        return cmd

    def submit(self, spec: JobSpec) -> str:
        command = self.command_for(spec)
        last_error: subprocess.CalledProcessError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                return result.stdout.strip().splitlines()[0].split(";")[0]
            except subprocess.CalledProcessError as exc:
                last_error = exc
                if attempt >= self.max_attempts or not _is_transient_sbatch_error(exc.stderr):
                    raise SlurmSubmissionError(command=command, exc=exc, attempts=attempt) from exc
                time.sleep(self.retry_delay_s * attempt)
        raise SlurmSubmissionError(command=command, exc=last_error, attempts=self.max_attempts)


class SlurmSubmissionError(RuntimeError):
    def __init__(
        self,
        *,
        command: list[str],
        exc: subprocess.CalledProcessError | None,
        attempts: int,
    ) -> None:
        self.command = command
        self.exc = exc
        self.attempts = attempts
        super().__init__(self._message())

    def _message(self) -> str:
        returncode = self.exc.returncode if self.exc else "unknown"
        lines = [
            f"sbatch failed with exit code {returncode} after {self.attempts} attempt(s).",
            f"Command: {shlex.join(self.command)}",
        ]
        if self.exc and self.exc.stderr.strip():
            lines.append(f"stderr: {self.exc.stderr.strip()}")
        if self.exc and self.exc.stdout.strip():
            lines.append(f"stdout: {self.exc.stdout.strip()}")
        return "\n".join(lines)


def _safe_job_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return cleaned[:128] or "segv2"


def _is_transient_sbatch_error(stderr: str | None) -> bool:
    if not stderr:
        return False
    text = stderr.lower()
    return any(
        fragment in text
        for fragment in (
            "unexpected message received",
            "socket timed out",
            "connection timed out",
            "temporarily unable",
            "temporarily unavailable",
            "slurm_receive_msg",
        )
    )
