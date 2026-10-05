"""Interactive Command-Line Interface (CLI) for Patient Appointment Scheduling Agent."""

from __future__ import annotations

import sys

from app.agent import SchedulingAgent

# Ensure UTF-8 output on Windows consoles if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_cli(agent: SchedulingAgent | None = None, debug: bool = False) -> None:
    """Run interactive terminal loop with a single stateful SchedulingAgent session."""
    print("🏥 2Care AI - The AI Receptionist for Healthcare ")
    print("Type 'exit' or 'quit' to end.\n")

    if agent is None:
        agent = SchedulingAgent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            response = agent.process_turn(user_input)

            if debug:
                safety = response.state.get("safety", {})
                tool_calls = response.tool_calls
                if tool_calls:
                    tools_str = " → ".join([t.get("tool", "tool") for t in tool_calls])
                    print(f"  [PIPELINE: USER → SAFETY ({safety.get('category', 'safe')}) → LLM → TOOL ({tools_str}) → SQLITE → LLM RESPONSE]")
                else:
                    print(f"  [PIPELINE: USER → SAFETY ({safety.get('category', 'safe')}) → LLM RESPONSE]")

            print(f"Agent: {response.response}\n")
        except Exception as err:
            print(f"Agent: An error occurred while processing your request: {err}\n")


def main() -> None:
    """Module entrypoint for uv run python -m app.cli."""
    debug_mode = "--debug" in sys.argv
    run_cli(debug=debug_mode)


if __name__ == "__main__":
    main()
