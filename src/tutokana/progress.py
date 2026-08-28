"""Progress reporting: a live bar on the console, detail lines in the log file.

The bar is written straight to the terminal and never goes through `logging`; the periodic
detail line is logged with `file_only=True` and filtered off the console by
`config.run_logging`. A log full of carriage returns is worthless, and a bar that scrolls
away is useless.

Rates are windowed rather than cumulative, so a slowdown is visible while it is happening,
and are withheld until the window spans `MIN_SPAN` seconds.
"""

from __future__ import annotations

import shutil
import sys
import time
from collections import deque


def format_duration(seconds: float) -> str:
    """`45s`, `12m30s`, `2h05m` — short enough to sit inside a one-line bar."""
    if seconds != seconds or seconds < 0 or seconds == float("inf"):  # nan/inf guard
        return "--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_rate(rate: float) -> str:
    """Items per second, or seconds per item once the former stops being readable."""
    if rate <= 0 or rate != rate:
        return "--"
    if rate >= 1.0:
        return f"{rate:.1f}/s"
    return f"{1.0 / rate:.1f}s each"


class Progress:
    """A console bar plus periodic file-only log lines.

    `every` controls the log line, not the bar: the bar redraws continuously (it costs
    nothing and is not recorded), while the log gets one detailed entry per `every` items so
    the file stays readable at any scale.
    """

    #: Rate and ETA are estimated over this many recent updates.
    WINDOW = 32
    #: Below this span the window is too short to divide by: one step a few microseconds
    #: after the previous one reads as a million per second, and the ETA that follows is
    #: worse than no ETA. Report "--" until there is enough elapsed time to mean something.
    MIN_SPAN = 0.5

    def __init__(
        self,
        total: int,
        logger,
        label: str,
        *,
        every: int = 10,
        unit: str = "it",
        start_at: int = 0,
        stream=None,
        enabled: bool | None = None,
    ):
        self.total = max(int(total), 0)
        self.logger = logger
        self.label = label
        self.every = max(int(every), 1)
        self.unit = unit
        self.stream = stream if stream is not None else sys.stdout
        # A bar is meaningless when the output is a pipe or a file — nohup, CI, `> out.txt` —
        # and would fill it with escape codes. The log line carries the same information.
        self.enabled = (
            enabled
            if enabled is not None
            else bool(getattr(self.stream, "isatty", lambda: False)())
        )
        # A resumed run starts part way along. The first mark has to carry that offset, or
        # the opening rate is computed as "500 steps since I started ten seconds ago".
        self.done = self._start_at = max(int(start_at), 0)
        self.started = time.time()
        self._marks: deque[tuple[float, int]] = deque(maxlen=self.WINDOW)
        self._marks.append((self.started, self.done))
        self._drawn = 0
        self._closed = False

    # -- statistics --------------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def rate(self) -> float:
        """Items per second over the recent window, not over the whole run."""
        if len(self._marks) < 2:
            return 0.0
        (t0, n0), (t1, n1) = self._marks[0], self._marks[-1]
        span = t1 - t0
        return (n1 - n0) / span if span >= self.MIN_SPAN else 0.0

    def eta(self) -> float:
        rate = self.rate()
        if rate <= 0 or self.total <= 0:
            return float("inf")
        return max(self.total - self.done, 0) / rate

    # -- rendering ---------------------------------------------------------------------

    def _bar(self, width: int) -> str:
        if self.total <= 0:
            return ""
        filled = int(width * self.done / self.total)
        return "█" * filled + "░" * (width - filled)

    def _line(self, suffix: str = "") -> str:
        pct = (100.0 * self.done / self.total) if self.total else 0.0
        counter = f"{self.done}/{self.total}" if self.total else str(self.done)
        tail = (
            f"{counter}  {pct:3.0f}%  {format_rate(self.rate())}  "
            f"elapsed {format_duration(self.elapsed)}  "
            f"eta {format_duration(self.eta())}{suffix}"
        )
        columns = shutil.get_terminal_size((100, 24)).columns
        budget = columns - len(self.label) - len(tail) - 6
        bar = self._bar(max(min(budget, 32), 0))
        return f"{self.label} {bar} {tail}" if bar else f"{self.label} {tail}"

    def _draw(self, suffix: str = "") -> None:
        if not self.enabled:
            return
        line = self._line(suffix)
        # Pad to erase whatever the previous, possibly longer, line left behind.
        self.stream.write("\r" + line.ljust(self._drawn))
        self.stream.flush()
        self._drawn = len(line)

    # -- driving -----------------------------------------------------------------------

    def update(self, n: int = 1, *, detail: str = "", suffix: str = "") -> None:
        """Advance by `n`. `suffix` rides on the bar; `detail` only on the log line."""
        self.done += n
        self._marks.append((time.time(), self.done))
        self._draw(suffix)
        if self.done % self.every == 0 or self.done >= self.total > 0:
            self.log_line(detail)

    def log_line(self, detail: str = "") -> None:
        counter = f"{self.done}/{self.total}" if self.total else str(self.done)
        self.logger.info(
            "[%s] %s  %s  elapsed %s  eta %s%s",
            self.label, counter, format_rate(self.rate()),
            format_duration(self.elapsed), format_duration(self.eta()),
            f"  {detail}" if detail else "",
            extra={"file_only": True},
        )

    def write(self, message: str) -> None:
        """Emit a message without the bar swallowing it or the bar being swallowed."""
        if self.enabled:
            self.stream.write("\r" + " " * self._drawn + "\r")
            self.stream.flush()
            self._drawn = 0
        self.logger.info("%s", message)
        self._draw()

    def close(self, detail: str = "") -> None:
        if self._closed:
            return
        self._closed = True
        if self.enabled:
            self.stream.write("\r" + " " * self._drawn + "\r")
            self.stream.flush()
            self._drawn = 0
        # Only what this process did: a resumed run must not claim credit for the steps it
        # inherited when reporting its own throughput.
        did = self.done - self._start_at
        rate = (did / self.elapsed) if self.elapsed > 0 else 0.0
        self.logger.info(
            "[%s] %d in %s (%s average)%s",
            self.label, did, format_duration(self.elapsed),
            format_rate(rate), f"  {detail}" if detail else "",
        )

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def memory_note(device) -> str:
    """Accelerator memory held, for the periodic log line. Empty when not applicable.

    Worth recording every time: the eval that ran for four hours did so because the working
    set grew with sequence length until the machine started swapping, and nothing in the log
    said so.
    """
    kind = getattr(device, "type", str(device))
    gb = 1024.0**3
    try:
        if kind == "mps":
            import torch

            return (
                f"mem {torch.mps.driver_allocated_memory() / gb:.1f}G held / "
                f"{torch.mps.current_allocated_memory() / gb:.1f}G live"
            )
        if kind == "cuda":
            import torch

            return (
                f"mem {torch.cuda.memory_reserved() / gb:.1f}G reserved / "
                f"{torch.cuda.memory_allocated() / gb:.1f}G live"
            )
    except Exception:  # a diagnostic must never be the thing that fails a run
        return ""
    return ""
