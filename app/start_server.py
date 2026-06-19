#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

APP_DIR = Path(__file__).resolve().parent
NODE_BIN_DIR = Path("/home/palodo2/node/bin")
NODE_BIN = NODE_BIN_DIR / "node"
NPM_BIN = NODE_BIN_DIR / "npm"
SERVER_JS = APP_DIR / "server.js"
FRONTEND_DIR = APP_DIR / "frontend"
FRONTEND_DIST = APP_DIR / "dist" / "index.html"
PID_FILE = APP_DIR / "server.pid"
LOG_FILE = APP_DIR / "server.log"
HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "5000"))
URL = f"http://{HOST}:{PORT}/"


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


def wait_for_http(timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(URL, timeout=0.8) as response:
                if 200 <= response.status < 500:
                    return True
        except URLError:
            time.sleep(0.25)
    return False


def tail_log(lines: int = 20) -> str:
    if not LOG_FILE.exists():
        return ""
    return "".join(LOG_FILE.read_text(errors="replace").splitlines(True)[-lines:])


def ensure_frontend_build() -> int:
    if FRONTEND_DIST.exists():
        return 0

    if not NPM_BIN.exists():
        print(f"npm binary not found: {NPM_BIN}", file=sys.stderr)
        return 1

    print("Frontend build not found. Building app/frontend before starting the server...")

    node_env = os.environ.copy()
    node_env["PATH"] = f"{NODE_BIN_DIR}:{node_env.get('PATH', '')}"

    install_cmd = [str(NPM_BIN), "ci"] if not (FRONTEND_DIR / "node_modules").exists() else None
    if install_cmd is not None:
        install_result = subprocess.run(
            install_cmd,
            cwd=FRONTEND_DIR,
            env=node_env,
            stdin=subprocess.DEVNULL,
        )
        if install_result.returncode != 0:
            print("Frontend dependency installation failed.", file=sys.stderr)
            return install_result.returncode

    build_result = subprocess.run(
        [str(NPM_BIN), "run", "build"],
        cwd=FRONTEND_DIR,
        env=node_env,
        stdin=subprocess.DEVNULL,
    )
    if build_result.returncode != 0:
        print("Frontend build failed.", file=sys.stderr)
    return build_result.returncode


def main() -> int:
    existing_pid = read_pid()
    if existing_pid and is_running(existing_pid):
        print(f"Server already running: {URL} (pid {existing_pid})")
        return 0
    if PID_FILE.exists():
        PID_FILE.unlink()

    if not NODE_BIN.exists():
        print(f"Node binary not found: {NODE_BIN}", file=sys.stderr)
        return 1
    if not SERVER_JS.exists():
        print(f"server.js not found: {SERVER_JS}", file=sys.stderr)
        return 1

    frontend_status = ensure_frontend_build()
    if frontend_status != 0:
        return frontend_status

    env = os.environ.copy()
    env["PATH"] = f"{NODE_BIN_DIR}:{env.get('PATH', '')}"

    with LOG_FILE.open("ab") as log_file:
        log_file.write(b"\n--- Starting ACL app server ---\n")
        process = subprocess.Popen(
            [str(NODE_BIN), str(SERVER_JS)],
            cwd=APP_DIR,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    PID_FILE.write_text(f"{process.pid}\n")

    if wait_for_http():
        print(f"Server started: {URL} (pid {process.pid})")
        print(f"Log file: {LOG_FILE}")
        return 0

    if process.poll() is not None:
        PID_FILE.unlink(missing_ok=True)
        print("Server failed to start. Last log lines:", file=sys.stderr)
        print(tail_log(), file=sys.stderr)
        return process.returncode or 1

    print(f"Server process started but HTTP check timed out: {URL} (pid {process.pid})")
    print(f"Check log file: {LOG_FILE}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
