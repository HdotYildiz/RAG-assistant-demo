from typer.testing import CliRunner

from technical_knowledge_assistant import cli
from technical_knowledge_assistant.models import Answer


def test_chat_exits_without_loading_a_second_question(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(cli, "_load_assistant", lambda: (object(), object()))

    result = runner.invoke(cli.app, ["chat"], input="exit\n")

    assert result.exit_code == 0
    assert "Type 'exit' or 'quit'" in result.output
    assert "Session ended." in result.output


def test_display_answer_shows_citations(capsys) -> None:
    answer = Answer(text="Use invoke().", citations=("freshstack-1",), sufficient_evidence=True)
    cli._display_answer(answer)

    assert capsys.readouterr().out == "Use invoke().\n\nSources:\n- freshstack-1\n"