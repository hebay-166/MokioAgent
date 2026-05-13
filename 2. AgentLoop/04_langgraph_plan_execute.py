from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

DEFAULT_TASK = "请检查 inbox，把 a.txt 移动到 archive，然后告诉我整理后的目录变化。"
WORKSPACE = Path(__file__).resolve().parent / "demo_workspace"


class AgentState(TypedDict):
    task: str
    plan: list[str]
    messages: list[BaseMessage]
    reflection: str


def reset_workspace() -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    (WORKSPACE / "inbox").mkdir(parents=True)
    (WORKSPACE / "archive").mkdir()
    (WORKSPACE / "inbox" / "a.txt").write_text(
        "Hello from MokioClaw AgentLoop demo.",
        encoding="utf-8",
    )


def show_workspace() -> str:
    items = sorted(WORKSPACE.rglob("*"))
    lines = []
    for item in items:
        rel = item.relative_to(WORKSPACE).as_posix()
        lines.append(f"- {rel}/" if item.is_dir() else f"- {rel}")
    return "\n".join(lines) or "(empty)"


def workspace_path(path: str) -> Path:
    path = path.strip().replace("\\", "/")
    return WORKSPACE if path == "." else WORKSPACE / path.strip("/")


@tool
def list_files(path: str = ".") -> str:
    """List files in the demo workspace."""
    target = workspace_path(path)
    if target.is_file():
        return target.relative_to(WORKSPACE).as_posix()
    files = sorted(item for item in target.rglob("*") if item.is_file())
    return "\n".join(f"- {item.relative_to(WORKSPACE).as_posix()}" for item in files) or "(empty)"


@tool
def move_file(source: str, target: str) -> str:
    """Move a file in the demo workspace."""
    source_path = workspace_path(source)
    target_path = workspace_path(target)
    if "." not in target_path.name:
        target_path = target_path / source_path.name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(target_path))
    return f"moved {source} -> {target_path.relative_to(WORKSPACE).as_posix()}"


def load_llm() -> ChatOpenAI:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return ChatOpenAI(
        model=os.getenv("MODEL", "qwen3.6-flash"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0,
    )


def parse_plan(text: str) -> list[str]:
    steps = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-*\d.、)]+\s*", "", line).strip()
        if line:
            steps.append(line)
    return steps or ["检查 inbox", "移动 a.txt 到 archive", "查看整理后的目录"]


PLANNER_PROMPT = """
把用户任务拆成 3 个步骤。
必须覆盖：检查 inbox、移动 a.txt 到 archive、查看整理后的目录。
每行一个步骤，不要写额外解释。
""".strip()

SYSTEM_PROMPT = """
你是一个 ReAct executor。
你会收到计划，请按计划用工具一步步完成任务。
每一轮最多调用一个工具。
如果有 reflection note，请优先参考它决定下一步。
""".strip()

REFLECTION_PROMPT = """
你是 reviewer。根据计划和最近工具结果，给 executor 一句下一步建议。
如果已经看到 archive/a.txt，请提醒 executor 停止调用工具并总结。
只输出一句 reflection note。
""".strip()


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_workspace()

    print("=== 04. LangGraph Plan + ReAct + Reflection ===")
    print("\n用户任务:")
    print(task)
    print("\n运行前 workspace:")
    print(show_workspace())

    tools = [list_files, move_file]
    tool_map = {item.name: item for item in tools}
    base_llm = load_llm()
    tool_llm = base_llm.bind_tools(tools)

    def planner_node(state: AgentState) -> AgentState:
        print("\n[planner] 生成计划")
        response = base_llm.invoke(
            [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=state["task"])]
        )

        plan = parse_plan(str(response.content))

        for index, step in enumerate(plan, start=1):
            print(f"{index}. {step}")

        plan_text = "\n".join(f"{index}. {step}" for index, step in enumerate(plan, start=1))

        return {
            **state,
            "plan": plan,
            "messages": [HumanMessage(content=f"用户任务：{state['task']}\n\n计划：\n{plan_text}")],
        }

    def agent_node(state: AgentState) -> AgentState:
        print("\n[agent] 按计划执行下一步")
        prompt = SYSTEM_PROMPT

        if state["reflection"]:
            prompt += f"\n\nReflection note: {state['reflection']}"

        response = tool_llm.invoke([SystemMessage(content=prompt), *state["messages"]])
        return {**state, "messages": [*state["messages"], response]}

    def tools_node(state: AgentState) -> AgentState:
        response = state["messages"][-1]
        new_messages = list(state["messages"])

        for tool_call in response.tool_calls:
            print("\n[tools] 执行工具")
            print(f"tool_name = {tool_call['name']}")
            print(f"tool_args = {tool_call['args']}")
            result = tool_map[tool_call["name"]].invoke(tool_call["args"])
            print(result)
            new_messages.append(
                ToolMessage(
                    content=str(result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {**state, "messages": new_messages}

    def reflection_node(state: AgentState) -> AgentState:
        print("\n[reflection] 复盘计划和工具结果")
        plan_text = "\n".join(state["plan"])
        transcript = "\n".join(str(message.content) for message in state["messages"][-4:])
        note = base_llm.invoke(
            [
                SystemMessage(content=REFLECTION_PROMPT),
                HumanMessage(content=f"计划：\n{plan_text}\n\n最近记录：\n{transcript}"),
            ]
        ).content
        print(note)
        return {**state, "reflection": str(note)}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("reflection", reflection_node)
    
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "reflection")
    graph.add_edge("reflection", "agent")

    result = graph.compile().invoke(
        {"task": task, "plan": [], "messages": [], "reflection": ""},
        config={"recursion_limit": 50},
    )

    print("\n最终回答:")
    print(result["messages"][-1].content)
    print("\n运行后 workspace:")
    print(show_workspace())


if __name__ == "__main__":
    main()
