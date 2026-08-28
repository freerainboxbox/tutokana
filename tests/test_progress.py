"""Progress reporting and the console/file split."""

from __future__ import annotations

import io
import logging

import pytest

from tutokana.config import FileOnly
from tutokana.progress import Progress, format_duration, format_rate, memory_note


class _Recorder(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def logger_and_records():
    log = logging.getLogger("tutokana.test.progress")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = _Recorder()
    log.addHandler(handler)
    yield log, handler.records
    log.handlers.clear()


def test_duration_formatting():
    assert format_duration(45) == "45s"
    assert format_duration(750) == "12m30s"
    assert format_duration(7500) == "2h05m"
    assert format_duration(float("inf")) == "--"
    assert format_duration(-1) == "--"


def test_rate_flips_to_seconds_per_item_when_slow():
    """0.067/s is unreadable; 15 s each is what you actually want to see."""
    assert format_rate(12.0) == "12.0/s"
    assert format_rate(0.067) == "14.9s each"
    assert format_rate(0) == "--"


def test_log_line_is_emitted_every_n_updates(logger_and_records):
    log, records = logger_and_records
    bar = Progress(100, log, "score", every=10, enabled=False)
    for _ in range(25):
        bar.update()
    # 10 and 20 only — the bar redraws every step but the log must not.
    assert len(records) == 2


def test_log_lines_are_marked_file_only(logger_and_records):
    log, records = logger_and_records
    bar = Progress(100, log, "score", every=5, enabled=False)
    for _ in range(5):
        bar.update()
    assert getattr(records[0], "file_only", False) is True


def test_file_only_records_are_filtered_from_the_console():
    """The bar owns the console; the detail line owns the log."""
    console = FileOnly()
    detail = logging.LogRecord("t", logging.INFO, "", 0, "detail", None, None)
    detail.file_only = True
    ordinary = logging.LogRecord("t", logging.INFO, "", 0, "ordinary", None, None)
    assert console.filter(detail) is False
    assert console.filter(ordinary) is True


def test_close_summary_reaches_the_console(logger_and_records):
    """The final line is not file_only — it is the one the operator wants to see."""
    log, records = logger_and_records
    bar = Progress(10, log, "score", every=100, enabled=False)
    bar.update(10)
    bar.close()
    assert getattr(records[-1], "file_only", False) is False


def test_resumed_run_does_not_report_a_fictitious_rate(logger_and_records):
    """start_at seeds the window, so the first rate is not '500 steps since I began'."""
    log, _ = logger_and_records
    bar = Progress(1954, log, "train", every=10, start_at=500, enabled=False)
    assert bar.done == 500
    bar.update()
    # The window is microseconds wide here; dividing by it would read as a million per
    # second. Honest answer is "not yet known".
    assert bar.rate() == 0.0
    assert format_rate(bar.rate()) == "--"


def test_rate_is_withheld_until_the_window_is_wide_enough():
    log = logging.getLogger("tutokana.test.progress.span")
    log.handlers.clear(); log.addHandler(logging.NullHandler())
    bar = Progress(1000, log, "score", every=10_000, enabled=False)
    bar._marks.clear()
    bar._marks.append((0.0, 0))
    bar._marks.append((Progress.MIN_SPAN / 2, 50))   # too short to trust
    assert bar.rate() == 0.0
    bar._marks.append((Progress.MIN_SPAN * 4, 100))  # wide enough now
    assert bar.rate() > 0.0


def test_resumed_run_reports_only_its_own_throughput(logger_and_records):
    log, records = logger_and_records
    bar = Progress(1954, log, "train", every=10_000, start_at=500, enabled=False)
    bar.update(20)
    bar.close()
    assert "20 in" in records[-1].getMessage()  # not 520


def test_eta_uses_the_recent_window_not_the_lifetime_average():
    """A cumulative average hides a throughput collapse; a window must not."""
    log = logging.getLogger("tutokana.test.progress.eta")
    log.handlers.clear(); log.addHandler(logging.NullHandler())
    bar = Progress(1000, log, "score", every=10_000, enabled=False)
    bar._marks.clear()
    bar._marks.append((0.0, 0))
    bar._marks.append((10.0, 100))     # 10/s over this window
    bar.done = 100
    assert bar.rate() == pytest.approx(10.0)
    assert bar.eta() == pytest.approx(90.0)


def test_bar_writes_to_the_stream_only_when_enabled():
    log = logging.getLogger("tutokana.test.progress.stream")
    log.handlers.clear(); log.addHandler(logging.NullHandler())

    quiet = io.StringIO()
    Progress(10, log, "score", stream=quiet, enabled=False).update()
    assert quiet.getvalue() == ""

    loud = io.StringIO()
    Progress(10, log, "score", stream=loud, enabled=True).update()
    assert "score" in loud.getvalue() and "\r" in loud.getvalue()


def test_bar_is_disabled_when_output_is_not_a_terminal():
    """Piping to a file must not fill it with carriage returns."""
    log = logging.getLogger("tutokana.test.progress.tty")
    log.handlers.clear(); log.addHandler(logging.NullHandler())
    assert Progress(10, log, "score", stream=io.StringIO()).enabled is False


def test_write_clears_the_bar_before_the_message(logger_and_records):
    log, records = logger_and_records
    stream = io.StringIO()
    bar = Progress(10, log, "score", stream=stream, enabled=True)
    bar.update()
    bar.write("[val] step 200")
    assert records[-1].getMessage() == "[val] step 200"
    assert getattr(records[-1], "file_only", False) is False


def test_zero_total_does_not_divide_by_zero(logger_and_records):
    log, _ = logger_and_records
    bar = Progress(0, log, "score", every=1, enabled=False)
    bar.update()
    bar.close()  # no ZeroDivisionError


def test_memory_note_never_raises(monkeypatch):
    """A diagnostic must not be the thing that fails a run."""
    import torch

    assert memory_note(torch.device("cpu")) == ""
    note = memory_note(torch.device("mps")) if torch.backends.mps.is_available() else ""
    assert isinstance(note, str)
