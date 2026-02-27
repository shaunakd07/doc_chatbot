from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional


class ManagedAppProcess:
    def __init__(
        self,
        command: List[str],
        *,
        cwd: Path,
        log_path: Path,
        startup_timeout_sec: float = 120.0,
    ) -> None:
        self.command = command
        self.cwd = cwd
        self.log_path = log_path
        self.startup_timeout_sec = max(5.0, float(startup_timeout_sec))
        self._proc: Optional[subprocess.Popen] = None
        self._log_handle = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, health_check: Callable[[], bool]) -> None:
        if self.is_running:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        self._proc = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        deadline = time.monotonic() + self.startup_timeout_sec
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"Managed app process exited with code {self._proc.returncode}")
            try:
                if health_check():
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError("Managed app did not pass health check before timeout.")

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(timeout=5.0)
                    except Exception:
                        pass
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

