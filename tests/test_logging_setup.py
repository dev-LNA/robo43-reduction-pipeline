from __future__ import annotations

import logging
import logging.handlers
import multiprocessing

import pytest

from robo43.logging_setup import ROOT_LOGGER_NAME, describe_handlers, setup_logging


@pytest.fixture(autouse=True)
def _restore_robo43_logger():
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    handlers = list(logger.handlers)
    level = logger.level
    propagate = logger.propagate
    yield
    for handler in logger.handlers:
        handler.close()
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_setup_logging_is_idempotent(tmp_path):
    logfile = tmp_path / "log.txt"
    setup_logging("INFO", logfile=logfile)
    setup_logging("INFO", logfile=logfile)
    setup_logging("INFO", logfile=logfile)

    info = describe_handlers()
    assert info["owned"] == 2
    assert info["n_handlers"] == 2

    logging.getLogger("robo43.sometest").info("hello")
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        handler.flush()

    content = logfile.read_text()
    assert content.count("hello") == 1


def test_setup_logging_default_is_idempotent_with_one_handler():
    setup_logging("INFO")
    setup_logging("INFO")
    setup_logging("INFO")
    info = describe_handlers()
    assert info["owned"] == 1
    assert info["n_handlers"] == 1


def test_level_string_sets_logger_level():
    setup_logging("DEBUG")
    assert logging.getLogger(ROOT_LOGGER_NAME).level == logging.DEBUG


def test_level_int_is_accepted():
    setup_logging(logging.WARNING)
    assert logging.getLogger(ROOT_LOGGER_NAME).level == logging.WARNING


def test_queue_path_installs_queue_handler():
    queue = multiprocessing.Queue()
    try:
        setup_logging("INFO", queue=queue)
        logger = logging.getLogger(ROOT_LOGGER_NAME)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.handlers.QueueHandler)
    finally:
        queue.close()
