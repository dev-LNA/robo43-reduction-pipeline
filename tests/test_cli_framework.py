from __future__ import annotations

import logging

import pytest

from robo43.cli.common import configure_from_args
from robo43.cli.main import build_parser, main
from robo43.cli.registry import Subcommand, all_subcommands, register
from robo43.exceptions import Robo43Error
from robo43.logging_setup import ROOT_LOGGER_NAME

_TEST_SUBCOMMAND_NAME = "_test_boom_robo43_cli_framework"


def _boom(_args) -> int:
    raise Robo43Error("boom")


if _TEST_SUBCOMMAND_NAME not in {s.name for s in all_subcommands()}:
    register(
        Subcommand(
            name=_TEST_SUBCOMMAND_NAME,
            help="test",
            add_arguments=lambda _parser: None,
            run=_boom,
        )
    )


@pytest.fixture(autouse=True)
def _reset_logging():
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    handlers = list(logger.handlers)
    level = logger.level
    propagate = logger.propagate
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


def test_unknown_subcommand_exits_two():
    with pytest.raises(SystemExit) as exc_info:
        main(["definitely-not-a-real-subcommand"])
    assert exc_info.value.code == 2


def test_robo43_error_yields_exit_two_no_traceback(capsys):
    assert main([_TEST_SUBCOMMAND_NAME]) == 2
    captured = capsys.readouterr()
    assert "error: boom" in captured.err
    assert "Traceback" not in captured.err


def test_pdb_off_by_default():
    args = build_parser().parse_args(["config", "path"])
    assert args.pdb is False


def test_verbose_and_loglevel_debug_resolve_same():
    verbose_args = build_parser().parse_args(["config", "path", "--verbose"])
    configure_from_args(verbose_args)
    assert logging.getLogger("robo43").level == logging.DEBUG

    loglevel_args = build_parser().parse_args(["config", "path", "--loglevel", "DEBUG"])
    configure_from_args(loglevel_args)
    assert logging.getLogger("robo43").level == logging.DEBUG


def test_strict_and_permissive_together_rejected(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["config", "path", "--strict", "--permissive"])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--strict" in captured.err
    assert "--permissive" in captured.err
