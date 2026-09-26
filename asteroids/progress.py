"""Visible progress for long local assays, without an optional dependency."""

from __future__ import annotations

import sys
import threading
import time


class ProgressBar:
    """Show phase/count/elapsed/ETA and keep ticking during blocking work."""

    def __init__(self, phase: str, total: int) -> None:
        if total < 1:
            raise ValueError("Progress total must be positive")
        self.phase = phase
        self.total = total
        self.current = 0
        self.started = 0.0
        self.last_report = 0.0
        self.terminal = sys.stdout.isatty()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.worker: threading.Thread | None = None

    def __enter__(self) -> "ProgressBar":
        self.started = time.monotonic()
        self.last_report = self.started
        self._report(force=True)
        self.worker = threading.Thread(target=self._heartbeat, daemon=True)
        self.worker.start()
        return self

    def _heartbeat(self) -> None:
        while not self.stop.wait(1.0):
            self._report()

    def _report(self, *, force: bool = False) -> None:
        with self.lock:
            now = time.monotonic()
            interval = 2.0 if self.terminal else 15.0
            if not force and now - self.last_report < interval:
                return
            self.last_report = now
            elapsed = max(now - self.started, 0.0)
            remaining = (
                f" ETA {int(elapsed / self.current * (self.total - self.current))}s"
                if self.current else " ETA calculating"
            )
            filled = int(24 * self.current / self.total)
            bar = "█" * filled + "░" * (24 - filled)
            line = (
                f"{self.phase} [{bar}] {self.current}/{self.total}"
                f" | {int(elapsed)}s elapsed{remaining}"
            )
            if self.terminal:
                sys.stdout.write("\r" + line + "\x1b[K")
            else:
                sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def advance(self) -> None:
        if self.current >= self.total:
            raise ValueError("Progress exceeded declared total")
        self.current += 1
        self._report(force=True)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=2.0)
        self._report(force=True)
        if self.terminal:
            sys.stdout.write("\n")
            sys.stdout.flush()
