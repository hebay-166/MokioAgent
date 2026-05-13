from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

DEFAULT_TASK = "请把 inbox 里的 a.txt 移动到 archive，然后生成一份简单的 Python 代码。"
WORKSPACE = Path(__file__).resolve().parent / "demo_workspace"


def reset_workspace() -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    (WORKSPACE / "inbox").mkdir(parents=True)
    (WORKSPACE / "archive").mkdir()
    (WORKSPACE / "inbox" / "a.txt").write_text(
        "Hello from MokioClaw MultiAgent demo.",
        encoding="utf-8",
    )


def workspace_path(path: str) -> Path:
    path = path.strip().replace("\\", "/")
    return WORKSPACE if path == "." else WORKSPACE / path.strip("/")


def show_workspace() -> str:
    items = sorted(WORKSPACE.rglob("*"))
    lines = []
    for item in items:
        rel = item.relative_to(WORKSPACE).as_posix()
        lines.append(f"- {rel}/" if item.is_dir() else f"- {rel}")
    return "\n".join(lines) or "(empty)"


@tool
def list_files(path: str = ".") -> str:
    """List files in the demo workspace."""
    target = workspace_path(path)
    files = sorted(item for item in target.rglob("*") if item.is_file())
    return "\n".join(f"- {item.relative_to(WORKSPACE).as_posix()}" for item in files) or "(empty)"


@tool
def move_file(source: str, target: str) -> str:
    """Move one file in the demo workspace."""
    source_path = workspace_path(source)
    target_path = workspace_path(target)
    if "." not in target_path.name:
        target_path = target_path / source_path.name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(target_path))
    return f"moved {source} -> {target_path.relative_to(WORKSPACE).as_posix()}"


@tool
def write_file(path: str, content: str) -> str:
    """Write a file in the demo workspace."""
    target_path = workspace_path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    return f"created {target_path.relative_to(WORKSPACE).as_posix()}"


def load_llm() -> ChatOpenAI:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return ChatOpenAI(
        model=os.getenv("MODEL", "qwen3.6-flash"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0,
    )


FILE_AGENT_PROMPT = """
你是 file_agent，只负责文件整理。
你可以使用 list_files 和 move_file。
目标：先查看 workspace，再把 inbox/a.txt 移动到 archive/a.txt，最后汇报目录变化。
不要写代码。
""".strip()

CODE_AGENT_PROMPT = """
你是 code_agent，只负责生成代码文件。
你可以使用 write_file。
请创建 check_archive.py，代码用于检查 archive/a.txt 是否存在。
不要移动文件。
""".strip()

SUPERVISOR_PROMPT = """
你是 supervisor agent，负责协调两个 specialist agent。

你不能直接整理文件，也不能直接写文件。
你只能调用这两个工具：
1. call_file_agent：让 file_agent 整理文件。
2. call_code_agent：让 code_agent 生成代码。

必须先调用 call_file_agent，再调用 call_code_agent，最后用中文总结两个 agent 的结果。
""".strip()


def last_message_text(result: dict) -> str:
    return str(result["messages"][-1].content)


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_workspace()

    print("=== 01. MultiAgent：Supervisor 把 Sub-Agent 当作 Tool ===")
    print("\n用户任务:")
    print(task)
    print("\n运行前 workspace:")
    print(show_workspace())

    llm = load_llm()

    file_agent = create_agent(
        llm,
        tools=[list_files, move_file],
        system_prompt=FILE_AGENT_PROMPT,
        name="file_agent",
    )
    code_agent = create_agent(
        llm,
        tools=[write_file],
        system_prompt=CODE_AGENT_PROMPT,
        name="code_agent",
    )

    @tool
    def call_file_agent(instruction: str) -> str:
        """Delegate a file-management task to file_agent."""
        print("\n[supervisor -> file_agent]")
        print(instruction)

        result = file_agent.invoke({"messages": [{"role": "user", "content": instruction}]})

        answer = last_message_text(result)
        print("\n[file_agent -> supervisor]")
        print(answer)

        return answer

    @tool
    def call_code_agent(instruction: str) -> str:
        """Delegate a code-generation task to code_agent."""
        print("\n[supervisor -> code_agent]")
        print(instruction)

        result = code_agent.invoke({"messages": [{"role": "user", "content": instruction}]})

        answer = last_message_text(result)

        print("\n[code_agent -> supervisor]")
        print(answer)
        return answer

    supervisor = create_agent(
        llm,
        tools=[call_file_agent, call_code_agent],
        system_prompt=SUPERVISOR_PROMPT,
        name="supervisor",
    )

    result = supervisor.invoke({"messages": [{"role": "user", "content": task}]})

    print("\n[supervisor] 最终回答:")
    print(last_message_text(result))
    print("\n运行后 workspace:")
    print(show_workspace())


if __name__ == "__main__":
    main()
