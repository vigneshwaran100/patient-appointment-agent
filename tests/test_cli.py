import os
import tempfile

import pytest

from app.agent import SchedulingAgent
from app.cli import run_cli
from app.prompts import clear_learned_policies
from app.storage.database import ClinicDatabase, get_db, reset_db_singleton


@pytest.fixture
def clean_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name

    db = ClinicDatabase(db_path=temp_path)
    db.seed_default_data()
    get_db(db_path=temp_path)
    clear_learned_policies()

    yield db

    reset_db_singleton()
    try:
        os.remove(temp_path)
    except OSError:
        pass


def test_cli_exit_command(monkeypatch, capsys):
    """Confirm CLI starts and exits immediately when typing 'exit'."""
    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    # Mock agent to avoid overhead in quick exit test
    class DummyAgent:
        def process_turn(self, msg: str):
            raise AssertionError("Agent should not be invoked on exit")

    run_cli(agent=DummyAgent())
    captured = capsys.readouterr()
    assert "🏥 Patient Appointment Scheduling Agent" in captured.out
    assert "Type 'exit' or 'quit' to end." in captured.out
    assert "Goodbye!" in captured.out


def test_cli_graceful_interrupt_handling(monkeypatch, capsys):
    """Confirm CLI handles KeyboardInterrupt and EOFError gracefully without crashing."""
    def raise_interrupt(prompt=""):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", raise_interrupt)

    class DummyAgent:
        pass

    run_cli(agent=DummyAgent())
    captured = capsys.readouterr()
    assert "Goodbye!" in captured.out


def test_cli_multi_turn_interaction(clean_db, monkeypatch, capsys):
    """Confirm CLI passes multiple turns into SchedulingAgent preserving state."""
    agent = SchedulingAgent()
    inputs = iter([
        "Hello, my name is Sarah Connor and my patient ID is P001. I need to see a cardiologist.",
        "quit"
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    run_cli(agent=agent)
    captured = capsys.readouterr()

    assert "🏥 Patient Appointment Scheduling Agent" in captured.out
    assert "Agent:" in captured.out
    # Multi-turn state verification
    assert agent.conversation_state["turn_count"] == 1
    assert agent.conversation_state["verification"]["is_verified"] is True
    assert "Goodbye!" in captured.out
