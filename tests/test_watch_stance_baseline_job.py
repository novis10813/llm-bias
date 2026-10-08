"""No network calls: fake execution or local scp through a mock SSH transport."""
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location(
    "watch_baseline", Path(__file__).parents[1] / "scripts/watch_stance_baseline_job.py")
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


def options(tmp_path, *extra):
    return watcher.parser().parse_args([
        "--host", "gpu-host", "--job-id", "baseline-1", "--remote-output", "runs/baseline",
        "--output-dir", str(tmp_path / "output"), "--report", str(tmp_path / "report.json"), *extra])


class FakeRun:
    def __init__(self, statuses, scp_code=0):
        self.statuses = iter(statuses)
        self.scp_code = scp_code
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        assert kwargs["timeout"] > 0
        assert command[1:9] == watcher.SSH_OPTIONS
        if command[0] == "scp":
            assert command[-2] == "gpu-host:runs/baseline"
            fetched = Path(command[-1], "baseline")
            fetched.mkdir()
            (fetched / "diagnostic.txt").write_text("partial or complete")
            return subprocess.CompletedProcess(command, self.scp_code, "", "copy error")
        assert command[-1] == "bash -l -s"
        assert kwargs["input"].startswith(f"echo {watcher._MARKER}\n")
        status = next(self.statuses)
        if isinstance(status, Exception):
            raise status
        if isinstance(status, tuple):
            return subprocess.CompletedProcess(command, status[0], status[1], "transport error")
        return subprocess.CompletedProcess(command, 0, watcher._MARKER + "\n" + status, "")


@pytest.mark.parametrize("terminal,remote,code", [
    ("EXIT 0", "succeeded", 0), ("EXIT 7", "failed", 1), ("MISSING", "lost", 1)])
def test_terminal_retrieves(tmp_path, terminal, remote, code):
    fake = FakeRun(["RUNNING", terminal])
    report, result = watcher.watch(options(tmp_path), run=fake, sleep=lambda _: None)
    assert result == code
    assert report["remote_status"] == remote
    assert report["retrieval_status"] == "succeeded"
    assert report["artifact_validation_status"] == "pending"
    assert report["research_eligible"] is False
    assert (tmp_path / "output/diagnostic.txt").exists()
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert len(fake.calls) == 3
    assert not Path(fake.calls[-1][0][-1]).exists()
    assert not (tmp_path / "output/baseline").exists()
    script = watcher.observation_script(options(tmp_path))
    assert 'tmux has-session -t "=$job"' in script
    assert script.index("tmux has-session") < script.index('elif [ -f "$dir/exit_code"')
    assert fake.calls[0][1]["timeout"] > 300


def test_recovery_resets_failures(tmp_path):
    fake = FakeRun([(255, ""), "garbage", "RUNNING", (255, ""), "EXIT 0"])
    delays = []
    report, code = watcher.watch(options(tmp_path), run=fake, sleep=delays.append)
    assert code == 0
    assert report["monitor_error"] is None
    assert delays == [5, 5, 5]


@pytest.mark.parametrize("failure", [
    (255, ""), "EXIT nope", "EXIT 1 2", "EXIT 0\nRUNNING",
    subprocess.TimeoutExpired("ssh", 400), OSError("unreachable")])
def test_exhaustion_is_not_lost(tmp_path, failure):
    fake = FakeRun([failure] * 3)
    report, code = watcher.watch(options(tmp_path), run=fake, sleep=lambda _: None)
    assert code == 1
    assert len(fake.calls) == 3
    assert report["remote_status"] == "unknown"
    assert report["remote_exit_code"] is None
    assert report["monitor_error"]
    assert report["retrieval_status"] == "not_attempted"


def test_failed_copy_preserves_staging(tmp_path):
    fake = FakeRun(["EXIT 0"], scp_code=1)
    report, code = watcher.watch(options(tmp_path), run=fake)
    assert code == 1
    assert report["retrieval_status"] == "failed"
    assert report["artifact_validation_status"] == "not_available"
    assert Path(report["staging_location"], "baseline/diagnostic.txt").exists()
    assert not (tmp_path / "output").exists()


def test_copy_timeout(tmp_path):
    fake = FakeRun(["MISSING"])
    def run(command, **kwargs):
        if command[0] == "scp":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return fake(command, **kwargs)
    report, code = watcher.watch(options(tmp_path), run=run)
    assert code == 1
    assert report["retrieval_status"] == "failed"
    assert Path(report["staging_location"]).exists()


@pytest.mark.parametrize("existing", ["output", "report.json"])
def test_existing_rejected_without_network(tmp_path, existing):
    target = tmp_path / existing
    target.write_text("keep")
    fake = FakeRun([])
    report, code = watcher.watch(options(tmp_path), run=fake)
    assert code == 1
    assert not fake.calls
    assert target.read_text() == "keep"
    assert report["monitor_error"]


def test_publication_race_never_overwrites(tmp_path):
    fake = FakeRun(["EXIT 0"])
    def run(command, **kwargs):
        result = fake(command, **kwargs)
        if command[0] == "scp":
            (tmp_path / "output").mkdir()
        return result
    report, code = watcher.watch(options(tmp_path), run=run)
    assert code == 1
    assert report["retrieval_status"] == "failed"
    assert list((tmp_path / "output").iterdir()) == []
    assert Path(report["staging_location"]).exists()


def test_report_write_failure_is_failure(tmp_path):
    args = options(tmp_path)
    args.report = tmp_path / "absent/report.json"
    report, code = watcher.watch(args, run=FakeRun(["EXIT 0"]))
    assert code == 1
    assert report["retrieval_status"] == "succeeded"
    assert report["report_error"]


@pytest.mark.parametrize("kind", ["missing", "file", "symlink"])
def test_invalid_fetched_child_preserves_container(tmp_path, kind):
    fake = FakeRun(["EXIT 0"])

    def run(command, **kwargs):
        if command[0] != "scp":
            return fake(command, **kwargs)
        container = Path(command[-1])
        (container / "partial.txt").write_text("keep")
        child = container / "baseline"
        if kind == "file":
            child.write_text("not a directory")
        elif kind == "symlink":
            child.symlink_to(tmp_path, target_is_directory=True)
        return subprocess.CompletedProcess(command, 0, "", "")

    report, code = watcher.watch(options(tmp_path), run=run)
    assert code == 1
    assert report["retrieval_status"] == "failed"
    assert "non-symlink directory" in report["retrieval_error"]
    assert Path(report["staging_location"], "partial.txt").read_text() == "keep"
    assert not (tmp_path / "output").exists()


def test_installed_legacy_scp_protocol(tmp_path):
    scp = shutil.which("scp")
    assert scp is not None, "installed scp is required for the protocol regression"
    remote_home = tmp_path / "remote-home"
    remote = remote_home / "runs/baseline"
    remote.mkdir(parents=True)
    (remote / "ordinary.txt").write_text("ordinary")
    (remote / ".hidden").write_text("dotfile")
    (remote / ".nested").mkdir()
    (remote / ".nested/child").write_text("nested")
    mock_ssh = tmp_path / "mock-ssh"
    mock_ssh.write_text('#!/bin/bash\ncd "$MOCK_REMOTE_HOME" || exit 1\nexec bash -c "${!#}"\n')
    mock_ssh.chmod(0o700)
    env = dict(os.environ, MOCK_REMOTE_HOME=str(remote_home))

    # Exercise actual legacy filename checking, not a fake copy implementation.
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    result = subprocess.run(
        [scp, "-O", "-S", str(mock_ssh), "-r", "gpu-host:runs/baseline/.", str(rejected)],
        capture_output=True, text=True, timeout=15, env=env)
    assert result.returncode != 0
    assert "unexpected filename: ." in result.stderr

    fake = FakeRun(["EXIT 0"])

    def run(command, **kwargs):
        if command[0] != "scp":
            return fake(command, **kwargs)
        assert "-T" not in command
        return subprocess.run([scp, "-O", "-S", str(mock_ssh), *command[1:]],
                              env=env, **kwargs)

    report, code = watcher.watch(options(tmp_path), run=run)
    assert code == 0, report
    assert report["staging_location"] is None
    output = tmp_path / "output"
    assert (output / "ordinary.txt").read_text() == "ordinary"
    assert (output / ".hidden").read_text() == "dotfile"
    assert (output / ".nested/child").read_text() == "nested"
    assert not (output / "baseline").exists()
    assert not list(tmp_path.glob(".output.staging-*"))


@pytest.mark.parametrize("flag,value", [
    ("--host", "a b"), ("--host", "-evil"), ("--job-id", "job;echo"),
    ("--job-id", "a\nb"), ("--remote-output", "../escape"),
    ("--remote-output", "/absolute"), ("--remote-output", "a b"),
    ("--remote-output", "a/$HOME"), ("--remote-output", "a/./b"),
    ("--check-interval", "nan"), ("--connection-retries", "0")])
def test_invalid_args(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        options(tmp_path, f"{flag}={value}")


@pytest.mark.parametrize("stdout", ["MISSING\n", "login banner\nEXIT 0\n"])
def test_missing_marker_is_monitor_error(tmp_path, stdout):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == "ssh"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    report, code = watcher.watch(options(tmp_path), run=run, sleep=lambda _: None)
    assert code == 1
    assert len(calls) == 3
    assert report["remote_status"] == "unknown"
    assert "missing observation script marker" in report["monitor_error"]
    assert report["retrieval_status"] == "not_attempted"


@pytest.mark.parametrize("tmux_code,stderr,expected", [
    (1, "no server running on /tmp/tmux-test/default", "lost"),
    (1, "can't find session: baseline-1", "lost"),
    (1, "error connecting to /tmp/tmux-test/default (No such file or directory)", "lost"),
    (127, "tmux: command not found", "unknown"),
    (1, "server exited unexpectedly", "unknown"),
    (2, "can't find session: baseline-1", "unknown"),
    (1, "permission denied", "unknown"),
])
def test_login_banner_and_tmux_outcomes(tmp_path, tmux_code, stderr, expected):
    import shlex

    fake = FakeRun([])
    observations = []

    def run(command, **kwargs):
        if command[0] == "scp":
            return fake(command, **kwargs)
        assert command == ["ssh", *watcher.SSH_OPTIONS, "gpu-host", "bash -l -s"]
        assert kwargs["input"] == f"echo {watcher._MARKER}\n{watcher.observation_script(args)}"
        observations.append(command)
        # Local login bash only: tmux is replaced, never contacts a server.
        prefix = ("printf 'login banner without newline'\n"
                  f"tmux() {{ printf '%s\\n' {shlex.quote(stderr)} >&2; return {tmux_code}; }}\n"
                  f"HOME={shlex.quote(str(tmp_path))}\n")
        return subprocess.run(["bash", "-l", "-s"],
                              **dict(kwargs, input=prefix + kwargs["input"]))

    args = options(tmp_path)
    report, code = watcher.watch(args, run=run, sleep=lambda _: None)
    assert code == 1
    assert report["remote_status"] == expected
    if expected == "unknown":
        assert len(observations) == args.connection_retries
        assert stderr in report["monitor_error"]
        assert report["retrieval_status"] == "not_attempted"
    else:
        assert len(observations) == 1
        assert report["monitor_error"] is None
        assert report["retrieval_status"] == "succeeded"
