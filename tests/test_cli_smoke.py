from __future__ import annotations

from typer.testing import CliRunner

from mokioclaw.cli.app import app


def test_cli_shows_help_without_task() -> None:
    runner = CliRunner()

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "mokioclaw" in result.output
