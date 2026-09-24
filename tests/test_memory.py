"""Tests for robo43.memory: /proc-based readings and the worker/wait helpers.

The capping/waiting logic is tested by monkeypatching the module's own
read_meminfo_available_mb rather than faking /proc files.
"""

from __future__ import annotations

import logging

import robo43.memory as memory


def test_read_rss_mb_is_positive() -> None:
    value = memory.read_rss_mb()
    assert value is not None
    assert value > 0.0


def test_read_meminfo_available_mb_is_positive() -> None:
    value = memory.read_meminfo_available_mb()
    assert value is not None
    assert value > 0.0


def test_release_memory_returns_sample_and_logs(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="robo43.memory"):
        sample = memory.release_memory("test-stage", level=logging.INFO)
    assert isinstance(sample, memory.MemorySample)
    assert any("release_memory[test-stage]" in r.message for r in caplog.records)


def test_safe_worker_count_caps_and_warns(monkeypatch, caplog) -> None:
    monkeypatch.setattr(memory, "read_meminfo_available_mb", lambda: 10000.0)
    with caplog.at_level(logging.WARNING, logger="robo43.memory"):
        result = memory.safe_worker_count(6, per_worker_mb=2000.0, reserve_mb=8000.0, stage="unit-test")
    assert result == 1
    assert any("safe_worker_count[unit-test]" in r.message for r in caplog.records)


def test_safe_worker_count_no_cap_needed(monkeypatch, caplog) -> None:
    monkeypatch.setattr(memory, "read_meminfo_available_mb", lambda: 100000.0)
    with caplog.at_level(logging.WARNING, logger="robo43.memory"):
        result = memory.safe_worker_count(4, per_worker_mb=1000.0, reserve_mb=8000.0)
    assert result == 4
    assert not caplog.records


def test_safe_worker_count_unreadable_meminfo_returns_requested(monkeypatch) -> None:
    monkeypatch.setattr(memory, "read_meminfo_available_mb", lambda: None)
    assert memory.safe_worker_count(5, per_worker_mb=1000.0, reserve_mb=8000.0) == 5


def test_safe_worker_count_always_at_least_one(monkeypatch) -> None:
    monkeypatch.setattr(memory, "read_meminfo_available_mb", lambda: 1000.0)
    result = memory.safe_worker_count(8, per_worker_mb=5000.0, reserve_mb=8000.0)
    assert result == 1


def test_wait_for_memory_returns_true_immediately(monkeypatch) -> None:
    monkeypatch.setattr(memory, "read_meminfo_available_mb", lambda: 20000.0)
    assert memory.wait_for_memory(required_mb=10000.0, stage="unit-test") is True


def test_wait_for_memory_times_out(monkeypatch, caplog) -> None:
    monkeypatch.setattr(memory, "read_meminfo_available_mb", lambda: 1000.0)
    with caplog.at_level(logging.WARNING, logger="robo43.memory"):
        result = memory.wait_for_memory(
            required_mb=10000.0, stage="unit-test", timeout_s=0.05, poll_interval_s=0.01
        )
    assert result is False
    assert any("timed out" in r.message for r in caplog.records)


def test_wait_for_memory_unreadable_returns_true(monkeypatch) -> None:
    monkeypatch.setattr(memory, "read_meminfo_available_mb", lambda: None)
    assert memory.wait_for_memory(required_mb=10000.0, stage="unit-test") is True


def test_log_peak_rss_summary_logs(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="robo43.memory"):
        memory.log_peak_rss_summary("unit-test-command")
    assert any("unit-test-command: peak RSS" in r.message for r in caplog.records)
