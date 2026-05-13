from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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
你是一个文件整理助手。
你可以反复调用工具，直到完成用户任务。
推荐顺序：先 list_files("inbox")，再 move_file("inbox/a.txt", "archive/a.txt")，再 list_files(".")，最后总结。
每一轮最多调用一个工具。
""".strip()


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_workspace()

    print("=== 01. 手写 while AgentLoop ===")
    print("\n用户任务:")
    print(task)
    print("\n运行前 workspace:")
    print(show_workspace())

    tools = [list_files, move_file]
    tool_map = {item.name: item for item in tools}
    llm = load_llm().bind_tools(tools)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=task)]

    for turn in range(1, 8):
        print(f"\n--- 第 {turn} 轮：模型思考 ---")
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print("\n最终回答:")
            print(response.content)
            break

        tool_call = response.tool_calls[0]
        print("\n模型决定调用工具:")
        print(f"tool_name = {tool_call['name']}")
        print(f"tool_args = {tool_call['args']}")

        result = tool_map[tool_call["name"]].invoke(tool_call["args"])
        print("\n工具返回:")
        print(result)

        messages.append(
            ToolMessage(
                content=str(result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    print("\n运行后 workspace:")
    print(show_workspace())


if __name__ == "__main__":
    main()
