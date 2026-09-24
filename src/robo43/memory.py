"""Explicit memory reclamation for the pipeline's heavy, long-running processes.

CPython frees Python-level objects immediately by reference counting, but a
large NumPy array's backing buffer is served from glibc's malloc arenas, and
glibc keeps freed arena memory around for reuse rather than handing it back to
the OS. Combined with reference cycles (WCS objects, HDULists, matplotlib
figures) that only a GC cycle -- not refcounting -- can break, a long-running
process can show RSS climbing steadily across frames even though nothing is
leaked in the Python sense. This module makes reclaiming that memory an
explicit, logged step instead of hoping the allocator does it on its own:
:func:`release_memory` runs ``gc.collect()`` to break cycles and then asks
glibc to return freed arena memory to the OS via ``malloc_trim(0)``.

This only touches memory this process's own allocator is holding. The
kernel's page cache (the "buff/cache" column in ``free``) is deliberately left
alone: it is memory the kernel already reclaims on demand under real pressure,
and force-dropping it needs root (``sync; echo 1 > /proc/sys/vm/drop_caches``)
-- a step this pipeline does not take and should not need to.

Linux-only: ``malloc_trim`` is a glibc extension with no equivalent this
module knows how to call elsewhere. On any other platform, or if libc cannot
be loaded, the trim step is silently skipped and only ``gc.collect()`` runs.
RSS and MemAvailable come from ``/proc/self/status`` and ``/proc/meminfo``
directly (no ``psutil`` dependency, which is not installed in this project).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import logging
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "STAGE_BYTES_PER_PIXEL",
    "MemorySample",
    "read_meminfo_available_mb",
    "read_peak_rss_mb",
    "read_rss_mb",
    "release_memory",
    "safe_worker_count",
    "wait_for_memory",
    "log_peak_rss_summary",
]

#: Documented bytes-per-raw-pixel multiplier for each heavy stage: every
#: full-resolution array (and its transient copies) that stage's per-frame
#: work holds at once, per pixel of the frame being processed. Used by
#: :func:`safe_worker_count` to turn "how many pixels is this frame" into
#: "how much RAM will one worker need". Calibrated against real T80S
#: photometry/astrometry runs on 9232x9216 (~85.2 Mpx) _proc.fits products
#: -- see this change's report for the measured peak-RSS-per-worker numbers
#: these were derived from. Treat as an approximation with headroom, not an
#: exact accounting: it must stay conservative (over- rather
#: under-estimate) because undercounting risks the OOM this module exists to
#: prevent.
STAGE_BYTES_PER_PIXEL: dict[str, float] = {
    "photometry": 96.0,
    "astrometry": 40.0,
    "reduce.science": 48.0,
}

_libc: ctypes.CDLL | None = None
if sys.platform.startswith("linux"):
    try:
        _libc_name = ctypes.util.find_library("c") or "libc.so.6"
        _libc = ctypes.CDLL(_libc_name)
    except OSError:
        _libc = None


def _malloc_trim() -> bool:
    """Ask glibc to return freed arena memory to the OS. No-op off Linux."""
    if _libc is None:
        return False
    try:
        _libc.malloc_trim(ctypes.c_size_t(0))
    except Exception:
        logger.debug("malloc_trim call failed", exc_info=True)
        return False
    return True


def _read_proc_field(path: str, field_name: str) -> float | None:
    """Read one ``field_name:  <value> kB`` line from a /proc file, in MB."""
    try:
        with open(path, encoding="utf-8") as fh:  # noqa: PTH123
            prefix = field_name + ":"
            for line in fh:
                if line.startswith(prefix):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except (OSError, IndexError, ValueError):
        return None
    return None


def read_rss_mb() -> float | None:
    """Current resident set size of this process, in MB, or None if unreadable."""
    return _read_proc_field("/proc/self/status", "VmRSS")


def read_peak_rss_mb() -> float | None:
    """Peak resident set size of this process (VmHWM), in MB, or None."""
    return _read_proc_field("/proc/self/status", "VmHWM")


def read_meminfo_available_mb() -> float | None:
    """System-wide estimate of memory available for new allocations, in MB."""
    return _read_proc_field("/proc/meminfo", "MemAvailable")


@dataclass(frozen=True, slots=True)
class MemorySample:
    """A paired (this process's RSS, system-wide MemAvailable) reading."""

    rss_mb: float | None
    available_mb: float | None


def _sample() -> MemorySample:
    return MemorySample(rss_mb=read_rss_mb(), available_mb=read_meminfo_available_mb())


def _fmt(value: float | None) -> float:
    return value if value is not None else -1.0


def release_memory(stage: str, *, level: int = logging.DEBUG) -> MemorySample:
    """Break reference cycles and return freed heap to the OS, logging before/after.

    ``stage`` is a short label naming the call site (e.g.
    ``"reduce.master_bias"``, ``"photometry:frame"``) for the log line only.
    Pass ``level=logging.INFO`` at genuine stage boundaries (the start/end of
    a pipeline phase); the default DEBUG suits finer-grained call sites, such
    as every Nth frame inside a serial loop.

    Runs unconditionally when called -- callers gate whether to call it at
    all on ``settings.runtime.release_memory``. Returns the post-collection
    :class:`MemorySample` (RSS/MemAvailable both None where /proc is
    unreadable, e.g. off Linux).
    """
    before = _sample()
    logger.log(
        level,
        "release_memory[%s] before: RSS=%.0f MB MemAvailable=%.0f MB",
        stage, _fmt(before.rss_mb), _fmt(before.available_mb),
    )
    collected = gc.collect()
    trimmed = _malloc_trim()
    after = _sample()
    logger.log(
        level,
        "release_memory[%s] after: RSS=%.0f MB MemAvailable=%.0f MB "
        "(gc_collected=%d malloc_trim=%s)",
        stage, _fmt(after.rss_mb), _fmt(after.available_mb), collected, trimmed,
    )
    return after


def wait_for_memory(
    required_mb: float,
    stage: str,
    timeout_s: float = 120.0,
    *,
    poll_interval_s: float = 5.0,
) -> bool:
    """Poll MemAvailable until it reaches ``required_mb`` or ``timeout_s`` elapses.

    Meant to run right before a RAM-heavy step, so a transient spike from
    another process sharing this machine (ollama, sync daemons) gets a
    chance to clear before this pipeline adds its own footprint on top. Never
    raises and never blocks past ``timeout_s``: on timeout it logs a WARNING
    and returns False so the caller proceeds anyway -- this is a courtesy
    wait for headroom, not the mechanism that prevents an OOM (that is
    :func:`safe_worker_count` capping this pipeline's own concurrency).

    Returns True once ``required_mb`` is available (immediately, or after
    waiting), False on timeout. When MemAvailable cannot be read at all
    (non-Linux, unreadable /proc), returns True immediately since there is
    nothing to wait on.
    """
    available = read_meminfo_available_mb()
    if available is None:
        return True
    if available >= required_mb:
        return True

    logger.warning(
        "wait_for_memory[%s]: MemAvailable=%.0f MB < required %.0f MB; "
        "waiting up to %.0fs for headroom",
        stage, available, required_mb, timeout_s,
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(max(0.0, min(poll_interval_s, deadline - time.monotonic())))
        available = read_meminfo_available_mb()
        if available is None or available >= required_mb:
            break

    if available is not None and available < required_mb:
        logger.warning(
            "wait_for_memory[%s] timed out after %.0fs: MemAvailable=%.0f MB "
            "still < required %.0f MB; proceeding anyway",
            stage, timeout_s, available, required_mb,
        )
        return False

    logger.info(
        "wait_for_memory[%s]: MemAvailable=%.0f MB, proceeding", stage, _fmt(available)
    )
    return True


def safe_worker_count(
    requested: int,
    per_worker_mb: float,
    reserve_mb: float,
    *,
    stage: str = "",
) -> int:
    """Cap ``requested`` workers so ``workers * per_worker_mb + reserve_mb <= MemAvailable``.

    Always returns at least 1: a single worker is allowed to proceed even
    when the budget is already tighter than that, since refusing to run at
    all is not this function's call to make. Logs a WARNING (naming
    ``stage``) whenever it caps the count down. When MemAvailable cannot be
    read, or ``per_worker_mb`` is not a usable positive estimate, there is
    nothing to cap against and ``requested`` (floored at 1) is returned
    unchanged.
    """
    requested = max(1, requested)
    if per_worker_mb <= 0:
        return requested
    available = read_meminfo_available_mb()
    if available is None:
        return requested

    budget_mb = available - reserve_mb
    max_workers = max(1, int(budget_mb // per_worker_mb)) if budget_mb > 0 else 1
    capped = min(requested, max_workers)
    if capped < requested:
        logger.warning(
            "safe_worker_count[%s]: capping workers %d -> %d "
            "(MemAvailable=%.0f MB, reserve=%.0f MB, per_worker=%.0f MB)",
            stage, requested, capped, available, reserve_mb, per_worker_mb,
        )
    return capped


def log_peak_rss_summary(command: str) -> None:
    """Log a one-line peak-RSS-for-this-process summary at a CLI command's exit."""
    logger.info("%s: peak RSS %.0f MB", command, _fmt(read_peak_rss_mb()))
