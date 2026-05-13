from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

DEFAULT_TASK = "请检查 inbox，把 a.txt 移动到 archive，然后告诉我整理后的目录变化。"
FILES: dict[str, str] = {}


def reset_workspace() -> None:
    FILES.clear()
    FILES["inbox/a.txt"] = "Hello from MokioClaw AgentLoop demo."


def show_workspace() -> str:
    return "\n".join(f"- {path}" for path in sorted(FILES)) or "(empty)"


@tool
def list_files(path: str = ".") -> str:
    """List files in the demo workspace."""
    prefix = "" if path == "." else path.strip("/") + "/"
    files = [name for name in sorted(FILES) if name.startswith(prefix)]
    return "\n".join(f"- {name}" for name in files) or "(empty)"


@tool
def move_file(source: str, target: str) -> str:
    """Move a file in the demo workspace."""
    content = FILES.pop(source)
    target_path = target if "." in Path(target).name else f"{target.rstrip('/')}/{Path(source).name}"
    FILES[target_path] = content
    return f"moved {source} -> {target_path}"


def load_llm() -> ChatOpenAI:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return ChatOpenAI(
        model=os.getenv("MODEL", "qwen3.6-flash"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0,
    )


SYSTEM_PROMPT = """
你是一个 ReAct 文件整理助手。
ReAct 的节奏是：观察当前状态 -> 选择一个工具行动 -> 查看工具结果 -> 决定下一步。
目标：检查 inbox，移动 inbox/a.txt 到 archive/a.txt，查看整理后的目录，然后总结。
每一轮最多调用一个工具。
""".strip()


def main() -> None:
    from langchain.agents import create_agent

    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_workspace()

    print("=== 02. LangChain create_agent：封装好的 ReAct Loop ===")
    print("\n用户任务:")
    print(task)
    print("\n运行前 workspace:")
    print(show_workspace())

    agent = create_agent(
        model=load_llm(),
        tools=[list_files, move_file],
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})

    print("\nAgent 消息流:")
    for message in result["messages"]:
        print(f"\n[{getattr(message, 'type', 'unknown')}]")
        if getattr(message, "tool_calls", None):
            print(message.tool_calls)
        elif getattr(message, "content", ""):
            print(message.content)

    print("\n运行后 workspace:")
    print(show_workspace())


if __name__ == "__main__":
    main()
