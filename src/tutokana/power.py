"""macOS battery State-of-Charge (SoC) guard for long batch training studies.

Only active on macOS hosts (`sys.platform == "darwin"`) that report a battery via
`pmset -g batt`. Linux/Windows hosts and desktop Macs without a battery return `None`
and bypass every wait cleanly.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Callable

#: Pause an active training run gracefully if SoC drops below this percentage after a step.
BATTERY_PAUSE_SOC = 40

#: Wait until SoC reaches this percentage before starting the next run or resuming a pause.
BATTERY_RESUME_SOC = 60

#: Seconds between SoC rechecks while waiting for the battery to recharge.
BATTERY_POLL_SECONDS = 5.0

#: Dedicated exit code (EX_TEMPFAIL) used by `train.py` when pausing for low battery so
#: `batch_study.py` can wait for recharge and resume automatically rather than stopping.
BATTERY_EXIT_CODE = 75

_BATTERY_LINE_RE = re.compile(r"^\s*-.*?\b(\d{1,3})%", re.MULTILINE)


def is_mac_host() -> bool:
    return sys.platform == "darwin"


def parse_pmset_soc(output: str) -> int | None:
    """Extract battery SoC percentage (0-100) from `pmset -g batt` output, or None."""
    match = _BATTERY_LINE_RE.search(output)
    if match is None:
        return None
    soc = int(match.group(1))
    return soc if 0 <= soc <= 100 else None


def read_battery_soc() -> int | None:
    """Return current battery SoC percentage on macOS hosts with a battery, else None."""
    if not is_mac_host():
        return None
    try:
        proc = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return parse_pmset_soc(proc.stdout)


def wait_for_battery(
    min_soc: int = BATTERY_RESUME_SOC,
    poll_seconds: float = BATTERY_POLL_SECONDS,
    *,
    read_soc: Callable[[], int | None] = read_battery_soc,
    sleep: Callable[[float], None] = time.sleep,
    reason: str = "",
    emit: Callable[[str], None] | None = None,
) -> int | None:
    """Block until reported battery SoC reaches `min_soc`, rechecking every `poll_seconds`.

    Returns the final observed SoC, or `None` if no battery is reported.
    """
    soc = read_soc()
    if soc is None or soc >= min_soc:
        return soc

    printer = emit or (lambda msg: print(msg, flush=True))
    why = f" ({reason})" if reason else ""
    printer(
        f"[battery] SoC is {soc}% (< {min_soc}%){why} — pausing until {min_soc}%, "
        f"rechecking every {poll_seconds:g}s..."
    )
    last_reported = soc
    while True:
        sleep(poll_seconds)
        soc = read_soc()
        if soc is None:
            printer("[battery] battery status unavailable — continuing.")
            return None
        if soc >= min_soc:
            printer(f"[battery] SoC reached {soc}% (>= {min_soc}%) — continuing.")
            return soc
        if soc != last_reported:
            printer(f"[battery] SoC is {soc}% (waiting for {min_soc}%)...")
            last_reported = soc
