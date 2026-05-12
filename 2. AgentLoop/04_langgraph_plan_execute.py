from __future__ import annotations

import re
import sys
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent_loop_common import (
    DEFAULT_TASK,
    TOOL_REGISTRY,
    load_llm_config,
    print_tool_call,
    print_workspace,
    reset_demo_workspace,
    safe_print,
)


class PlanState(TypedDict):
    task: str
    plan: list[str]
    current_step: int
    observations: list[str]
    final_answer: str


PLANNER_PROMPT = """
把用户任务拆成 3 个可执行步骤。
必须覆盖：检查 inbox、移动 a.txt 到 archive、查看整理后的目录。
每行一个步骤，不要写额外解释。
""".strip()

REVIEWER_PROMPT = """
你是 replanner/reviewer。
根据计划和执行观察，判断是否完成；如果完成，用中文给出简短最终总结。
""".strip()


def parse_plan(text: str) -> list[str]:
    steps = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-*\d.、)]+\s*", "", line).strip()
        if line:
            steps.append(line)
    return steps or ["检查 inbox", "移动 inbox/a.txt 到 archive/a.txt", "查看整理后的目录"]


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_demo_workspace()

    safe_print("=== 04. LangGraph Plan-and-Execute ===")
    safe_print("\n用户任务:")
    safe_print(task)
    print_workspace("运行前的 demo workspace")

    llm = ChatOpenAI(**load_llm_config(), temperature=0)

    def planner_node(state: PlanState) -> PlanState:
        safe_print("\n[planner] 生成计划")
        response = llm.invoke(
            [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=state["task"])]
        )
        plan = parse_plan(str(response.content))
        for index, step in enumerate(plan, start=1):
            safe_print(f"{index}. {step}")
        return {**state, "plan": plan}

    def executor_node(state: PlanState) -> PlanState:
        step = state["plan"][state["current_step"]]
        safe_print(f"\n[executor] 执行步骤 {state['current_step'] + 1}: {step}")

        observations = list(state["observations"])

        if state["current_step"] == 0:
            tool_call = {"name": "list_files", "args": {"path": "inbox"}}
        elif state["current_step"] == 1:
            tool_call = {
                "name": "move_file",
                "args": {"source": "inbox/a.txt", "target": "archive/a.txt"},
            }
        else:
            tool_call = {"name": "list_files", "args": {"path": "."}}

        print_tool_call(tool_call)
        result = TOOL_REGISTRY[tool_call["name"]].invoke(tool_call["args"])
        safe_print(result)
        observations.append(f"{step} -> {result}")

        return {**state, "observations": observations}

    def reviewer_node(state: PlanState) -> PlanState:
        next_step = state["current_step"] + 1
        if next_step < len(state["plan"]):
            safe_print("\n[reviewer] 当前步骤完成，继续下一步")
            return {**state, "current_step": next_step}

        safe_print("\n[reviewer] 计划完成，生成最终总结")
        response = llm.invoke(
            [
                SystemMessage(content=REVIEWER_PROMPT),
                HumanMessage(content="\n".join(state["observations"])),
            ]
        )
        safe_print(response.content)
        return {**state, "current_step": next_step, "final_answer": str(response.content)}

    def should_continue(state: PlanState) -> str:
        return END if state["final_answer"] else "executor"

    graph = StateGraph(PlanState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges("reviewer", should_continue)

    graph.compile().invoke(
        {
            "task": task,
            "plan": [],
            "current_step": 0,
            "observations": [],
            "final_answer": "",
        },
        config={"recursion_limit": 12},
    )
    print_workspace("运行后的 demo workspace")


if __name__ == "__main__":
    main()
