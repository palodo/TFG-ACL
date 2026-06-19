#!/usr/bin/env python3
import os
import signal
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PID_FILE = APP_DIR / "server.pid"
LOG_FILE = APP_DIR / "server.log"


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid():
    try:
        return int(PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def wait_until_stopped(pid: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_running(pid):
            return True
        time.sleep(0.2)
    return not is_running(pid)


def main() -> int:
    pid = read_pid()
    if not pid:
        print("No server.pid found. Nothing to stop.")
        return 0

    if not is_running(pid):
        PID_FILE.unlink(missing_ok=True)
        print(f"Stale pid file removed: {pid}")
        return 0

    print(f"Stopping server pid {pid}...")
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        print("Server was already stopped.")
        return 0
    except PermissionError:
        os.kill(pid, signal.SIGTERM)

    if not wait_until_stopped(pid, 6.0):
        print("Server did not exit after SIGTERM; sending SIGKILL...")
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            os.kill(pid, signal.SIGKILL)
        wait_until_stopped(pid, 2.0)

    PID_FILE.unlink(missing_ok=True)
    with LOG_FILE.open("ab") as log_file:
        log_file.write(b"--- Stopped ACL app server ---\n")
    print("Server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
