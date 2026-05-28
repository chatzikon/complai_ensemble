import os
import signal
import subprocess
from pathlib import Path
from typing import Iterator


RUNNING_PID_FILE = Path("/tmp/complai_running_eval.pid")


def stream_command(cmd: list[str]) -> tuple[Iterator[str], subprocess.Popen]:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    RUNNING_PID_FILE.write_text(str(process.pid), encoding="utf-8")

    def iterator():
        assert process.stdout is not None

        for line in process.stdout:
            yield line

    return iterator(), process


def kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            pass


def kill_running_evaluation() -> bool:
    if not RUNNING_PID_FILE.exists():
        return False

    try:
        pid = int(RUNNING_PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        RUNNING_PID_FILE.unlink(missing_ok=True)
        return False

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        RUNNING_PID_FILE.unlink(missing_ok=True)
        return True
    except ProcessLookupError:
        RUNNING_PID_FILE.unlink(missing_ok=True)
        return False
    except Exception:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            RUNNING_PID_FILE.unlink(missing_ok=True)
            return True
        except Exception:
            return False


def clear_running_evaluation_pid(process: subprocess.Popen | None = None) -> None:
    if not RUNNING_PID_FILE.exists():
        return

    if process is None:
        RUNNING_PID_FILE.unlink(missing_ok=True)
        return

    try:
        stored_pid = int(RUNNING_PID_FILE.read_text(encoding="utf-8").strip())

        if stored_pid == process.pid:
            RUNNING_PID_FILE.unlink(missing_ok=True)

    except Exception:
        RUNNING_PID_FILE.unlink(missing_ok=True)