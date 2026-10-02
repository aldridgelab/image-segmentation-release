from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deciphaer_image_segmentation.slurm import (
    JobSpec,
    SlurmSubmissionError,
    SlurmSubmitter,
    _is_transient_sbatch_error,
    _safe_job_name,
)


def test_safe_job_name_replaces_unsafe_chars_and_truncates() -> None:
    assert _safe_job_name("segv2_batch1/momia") == "segv2_batch1_momia"
    assert _safe_job_name("a b c!") == "a_b_c_"
    long = "x" * 200
    assert _safe_job_name(long) == "x" * 128
    assert _safe_job_name("!!!") == "_"  # non-empty fallback


def test_is_transient_sbatch_error_detects_known_fragments() -> None:
    assert _is_transient_sbatch_error("sbatch: error: socket timed out on send/recv")
    assert _is_transient_sbatch_error("Unexpected message received")
    assert _is_transient_sbatch_error("slurmctld temporarily unavailable")
    assert not _is_transient_sbatch_error(None)
    assert not _is_transient_sbatch_error("")
    assert not _is_transient_sbatch_error("Invalid partition specified")


def test_command_for_builds_expected_flags() -> None:
    spec = JobSpec(
        name="segv2_b1/momia",
        stage="momia",
        script=Path("/runs/x.sh"),
        args=("--config", "cfg.yaml"),
        dependencies=("123", "456"),
        partition="preempt",
        account="lab",
        time="08:00:00",
        mem="16g",
        cpus_per_task=4,
        output_path="/logs/out.log",
        error_path="/logs/err.log",
        gres="gpu:1",
        requeue=True,
    )
    cmd = SlurmSubmitter().command_for(spec)
    assert cmd[0] == "sbatch"
    assert "--parsable" in cmd
    assert "--job-name=segv2_b1_momia" in cmd
    assert "--partition=preempt" in cmd
    assert "--account=lab" in cmd
    assert "--time=08:00:00" in cmd
    assert "--mem=16g" in cmd
    assert "--cpus-per-task" in cmd and "4" in cmd
    assert "--gres=gpu:1" in cmd
    assert "--requeue" in cmd
    assert "--output=/logs/out.log" in cmd
    assert "--error=/logs/err.log" in cmd
    assert "--dependency=afterok:123:456" in cmd
    assert cmd[-3:] == ["/runs/x.sh", "--config", "cfg.yaml"]


def test_command_for_uses_dependency_type_when_set() -> None:
    spec = JobSpec(
        name="watchdog",
        stage="compile",
        script=Path("x.sh"),
        dependencies=("100",),
        dependency_type="afterany",
    )
    cmd = SlurmSubmitter().command_for(spec)
    assert "--dependency=afterany:100" in cmd


def test_submit_returns_first_token_from_sbatch_output() -> None:
    spec = JobSpec(name="j", stage="s", script=Path("x.sh"))
    completed = subprocess.CompletedProcess([], 0, stdout="12345;cluster\n", stderr="")
    with patch("subprocess.run", return_value=completed):
        job_id = SlurmSubmitter().submit(spec)
    assert job_id == "12345"


def test_submit_retries_on_transient_error_then_succeeds() -> None:
    spec = JobSpec(name="j", stage="s", script=Path("x.sh"))
    transient = subprocess.CalledProcessError(
        returncode=1, cmd=["sbatch"], stderr="socket timed out"
    )
    ok = subprocess.CompletedProcess([], 0, stdout="999\n", stderr="")
    sleep = MagicMock()
    with patch("subprocess.run", side_effect=[transient, ok]), patch(
        "time.sleep", sleep
    ):
        job_id = SlurmSubmitter(max_attempts=3, retry_delay_s=0.0).submit(spec)
    assert job_id == "999"
    sleep.assert_called()


def test_submit_raises_immediately_on_non_transient_error() -> None:
    spec = JobSpec(name="j", stage="s", script=Path("x.sh"))
    fatal = subprocess.CalledProcessError(
        returncode=1, cmd=["sbatch"], stderr="Invalid partition"
    )
    fatal.stdout = ""
    with patch("subprocess.run", side_effect=fatal):
        with pytest.raises(SlurmSubmissionError) as excinfo:
            SlurmSubmitter(max_attempts=3, retry_delay_s=0.0).submit(spec)
    assert excinfo.value.attempts == 1
    assert "Invalid partition" in str(excinfo.value)


def test_submit_gives_up_after_max_attempts() -> None:
    spec = JobSpec(name="j", stage="s", script=Path("x.sh"))
    transient = subprocess.CalledProcessError(
        returncode=1, cmd=["sbatch"], stderr="socket timed out"
    )
    transient.stdout = ""
    with patch("subprocess.run", side_effect=transient), patch("time.sleep"):
        with pytest.raises(SlurmSubmissionError) as excinfo:
            SlurmSubmitter(max_attempts=2, retry_delay_s=0.0).submit(spec)
    assert excinfo.value.attempts == 2
