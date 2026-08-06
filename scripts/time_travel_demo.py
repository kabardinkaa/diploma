from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.agent_persistent import build_agent


class DemoModel:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, config=None):
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="Запрос на отправку обработан.")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "send_telegram_message",
                    "args": {
                        "chat_id": "demo-support",
                        "text": "Срочно нужна помощь с VPN",
                    },
                    "id": "demo-send-call",
                    "type": "tool_call",
                }
            ],
        )


async def demo_send(payload: dict[str, str]) -> str:
    print(f"SIDE EFFECT: [TELEGRAM -> {payload['chat_id']}] {payload['text']}")
    return f"Сообщение отправлено в {payload['chat_id']}"


def config(thread_id: str, checkpoint_id: str | None = None) -> dict:
    configurable = {"thread_id": thread_id}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def initial_input() -> dict:
    return {
        "messages": [("user", "Отправь подтвержденное сообщение в поддержку")],
        "user_role": "write-with-approve",
        "tool_results": [],
    }


async def run_branch(graph, thread_id: str, decision: bool) -> dict:
    run_config = config(thread_id)
    interrupted = await graph.ainvoke(initial_input(), config=run_config)
    print(f"\n{thread_id} __interrupt__: {interrupted['__interrupt__']}")
    result = await graph.ainvoke(Command(resume=decision), config=run_config)
    print(f"{thread_id} resume={decision}: sent={result['sent']}")
    return result


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agent-time-travel-") as directory:
        db_path = str(Path(directory) / "checkpoints.db")
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            await saver.setup()
            graph = build_agent(saver, model=DemoModel(), send_handler=demo_send)

            history_config = config("history-demo")
            interrupted = await graph.ainvoke(initial_input(), config=history_config)
            print("__interrupt__ payload:")
            print(interrupted["__interrupt__"])

            history = [item async for item in graph.aget_state_history(history_config)]
            print("\nCheckpoint history:")
            print("checkpoint_id | next")
            for snapshot in history:
                checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
                print(f"{checkpoint_id} | {snapshot.next}")

            before_send = next(
                item
                for item in history
                if item.next == ("confirm_and_execute_send_telegram_message",)
            )
            checkpoint_id = before_send.config["configurable"]["checkpoint_id"]
            past = await graph.aget_state(config("history-demo", checkpoint_id))
            print("\nPast checkpoint before send:")
            print(f"checkpoint_id={checkpoint_id}")
            print(f"preview={past.values['pending_send']}")
            print(f"sent={past.values['sent']}")
            print(f"next={past.next}")

            rejected = await run_branch(graph, "branch-reject", False)
            approved = await run_branch(graph, "branch-approve", True)
            print("\nBranch summary:")
            print(f"branch-reject sent={rejected['sent']}")
            print(f"branch-approve sent={approved['sent']}")


if __name__ == "__main__":
    asyncio.run(main())
