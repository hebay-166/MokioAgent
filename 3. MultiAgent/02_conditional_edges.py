from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph

DEFAULT_TASK = "请把 inbox 里的 a.txt 移动到 archive，然后生成一份简单的 Python 代码。"
WORKSPACE = Path(__file__).resolve().parent / "demo_workspace"


class MultiAgentState(TypedDict):
    task: str
    next_agent: str
    file_report: str
    code_report: str
    final_answer: str


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


SUPERVISOR_PROMPT = """
你是 supervisor agent。
你负责决定下一个应该执行的 agent。

可选 agent：
- file_agent：整理文件
- code_agent：生成代码
- finish：所有任务完成

只输出一个词：file_agent、code_agent 或 finish。
""".strip()

SUMMARY_PROMPT = """
你是 supervisor agent。
根据 file_agent 和 code_agent 的报告，用中文做一个简短总结。
""".strip()

FILE_AGENT_PROMPT = """
你是 file_agent，只负责文件整理。
你可以使用 list_files 和 move_file。
你需要根据用户任务自行决定要查看哪些目录、移动哪些文件，并汇报目录变化。
不要写代码。
""".strip()

CODE_AGENT_PROMPT = """
你是 code_agent，只负责生成代码文件。
你可以使用 write_file。
你需要根据用户任务和 file_agent 的报告，自行决定要生成什么代码文件。
不要移动文件。
""".strip()


def last_message_text(result: dict) -> str:
    return str(result["messages"][-1].content)


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_workspace()

    print("=== 02. MultiAgent：每个 Agent 是一个 Node，用 conditional_edge 路由 ===")
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

    def supervisor_node(state: MultiAgentState) -> MultiAgentState:
        print("\n[supervisor] 决定下一个 agent")

        if not state["file_report"]:
            next_agent = "file_agent"
        elif not state["code_report"]:
            next_agent = "code_agent"
        else:
            next_agent = "finish"

        decision = llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(
                    content=(
                        f"用户任务：{state['task']}\n\n"
                        f"file_report：{state['file_report'] or '(empty)'}\n\n"
                        f"code_report：{state['code_report'] or '(empty)'}\n\n"
                        f"请判断下一个 agent。建议答案：{next_agent}"
                    )
                ),
            ]
        ).content

        decision = str(decision).strip()
        if decision not in {"file_agent", "code_agent", "finish"}:
            decision = next_agent
        print(decision)

        if decision == "finish":
            final_answer = llm.invoke(
                [
                    SystemMessage(content=SUMMARY_PROMPT),
                    HumanMessage(
                        content=(
                            f"file_agent:\n{state['file_report']}\n\n"
                            f"code_agent:\n{state['code_report']}"
                        )
                    ),
                ]
            ).content
            return {**state, "next_agent": "finish", "final_answer": str(final_answer)}

        return {**state, "next_agent": decision}

    def file_agent_node(state: MultiAgentState) -> MultiAgentState:
        print("\n[file_agent node] 调用真正的 file_agent")
        result = file_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"用户原始任务：{state['task']}\n\n"
                            "你是 file_agent，只处理其中和文件整理有关的部分。"
                        ),
                    }
                ]
            }
        )
        report = last_message_text(result)
        print(report)
        return {**state, "file_report": report}

    def code_agent_node(state: MultiAgentState) -> MultiAgentState:
        print("\n[code_agent node] 调用真正的 code_agent")
        result = code_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"用户原始任务：{state['task']}\n\n"
                            f"file_agent 已完成的结果：\n{state['file_report']}\n\n"
                            "你是 code_agent，只处理其中和代码生成有关的部分。"
                        ),
                    }
                ]
            }
        )
        report = last_message_text(result)
        print(report)
        return {**state, "code_report": report}

    def route_next(state: MultiAgentState) -> str:
        if state["next_agent"] == "file_agent":
            return "file_agent"
        if state["next_agent"] == "code_agent":
            return "code_agent"
        return END

    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("file_agent", file_agent_node)
    graph.add_node("code_agent", code_agent_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_next)
    graph.add_edge("file_agent", "supervisor")
    graph.add_edge("code_agent", "supervisor")

    result = graph.compile().invoke(
        {
            "task": task,
            "next_agent": "",
            "file_report": "",
            "code_report": "",
            "final_answer": "",
        }
    )

    print("\n[supervisor] 最终回答:")
    print(result["final_answer"])
    print("\n运行后 workspace:")
    print(show_workspace())


if __name__ == "__main__":
    main()
