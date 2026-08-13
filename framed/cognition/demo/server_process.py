"""Own a local HTTP subprocess for restart-durability checks."""

from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional


def wait_for_http(url: str, *, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(0.2)
    return False


def wait_for_http_gone(url: str, *, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            return True
        time.sleep(0.2)
    return False


@dataclass
class ManagedServer:
    command: List[str]
    cwd: Path
    env: Dict[str, str]
    url: str
    log_path: Path
    health_path: str = "/upload"
    proc: Optional[subprocess.Popen] = field(default=None, init=False)
    log_handle: Optional[object] = field(default=None, init=False)

    @property
    def pid(self) -> Optional[int]:
        if self.proc is None or self.proc.poll() is not None:
            return None
        return self.proc.pid

    @property
    def health_url(self) -> str:
        return self.url.rstrip("/") + self.health_path

    def start(self, *, timeout: float = 90.0) -> int:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = open(self.log_path, "ab")
        popen_kwargs: Dict[str, object] = {
            "cwd": str(self.cwd),
            "env": self.env,
            "stdout": self.log_handle,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(self.command, **popen_kwargs)
        if not wait_for_http(self.health_url, timeout=timeout):
            self.stop()
            raise RuntimeError(f"Server failed to become healthy at {self.health_url}")
        assert self.proc.pid is not None
        return self.proc.pid

    def stop(self, *, timeout: float = 20.0) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        wait_for_http_gone(self.health_url, timeout=timeout)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        self.proc = None

    def restart(self, *, timeout: float = 90.0) -> tuple[int, int]:
        old_pid = self.pid
        if old_pid is None:
            raise RuntimeError("Cannot restart: server is not running")
        self.stop(timeout=min(timeout, 20.0))
        new_pid = self.start(timeout=timeout)
        if new_pid == old_pid:
            raise RuntimeError(f"Restart did not change PID ({old_pid})")
        return old_pid, new_pid


def flask_env(base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    env = dict(base or os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["FRAMED_COGNITION_V1"] = "true"
    return env
