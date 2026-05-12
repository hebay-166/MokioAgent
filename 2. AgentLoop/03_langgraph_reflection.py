from __future__ import annotations

import sys
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent_loop_common import (
    DEFAULT_TASK,
    TOOL_REGISTRY,
    TOOLS,
    load_llm_config,
    print_tool_call,
    print_workspace,
    reset_demo_workspace,
    safe_print,
)


class AgentState(TypedDict):
    messages: list[BaseMessage]
    reflection: str


SYSTEM_PROMPT = """
你是一个文件整理助手。
你可以循环调用工具完成任务。
目标固定为：检查 inbox，移动 inbox/a.txt 到 archive/a.txt，查看整理后的目录，然后总结。
不要读取文件内容，不要做目标之外的检查。
如果有 reflection note，请用它修正下一步动作。
""".strip()


REFLECTION_PROMPT = """
你是 reviewer。只根据固定目标给下一步建议。
目标：检查 inbox -> 移动 inbox/a.txt 到 archive/a.txt -> 查看整理后的目录 -> 总结。
如果已经看到 archive/a.txt，请提醒 agent 停止调用工具并总结。
不要调用工具，只输出 reflection note。
""".strip()


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_demo_workspace()

    safe_print("=== 03. LangGraph + Reflection 节点 ===")
    safe_print("\n用户任务:")
    safe_print(task)
    print_workspace("运行前的 demo workspace")

    llm = ChatOpenAI(**load_llm_config(), temperature=0)
    tool_llm = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> AgentState:
        prompt = SYSTEM_PROMPT
        if state["reflection"]:
            prompt += f"\n\nReflection note: {state['reflection']}"

        safe_print("\n[agent] 模型思考")
        response = tool_llm.invoke([SystemMessage(content=prompt), *state["messages"]])
        return {"messages": [*state["messages"], response], "reflection": state["reflection"]}

    def tool_node(state: AgentState) -> AgentState:
        response = state["messages"][-1]
        new_messages = list(state["messages"])

        for tool_call in response.tool_calls:
            safe_print("\n[tools] 执行工具")
            print_tool_call(tool_call)
            result = TOOL_REGISTRY[tool_call["name"]].invoke(tool_call["args"])
            safe_print(result)
            new_messages.append(
                ToolMessage(
                    content=str(result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": new_messages, "reflection": state["reflection"]}

    def reflection_node(state: AgentState) -> AgentState:
        safe_print("\n[reflection] 复盘工具结果")
        transcript = "\n".join(str(message.content) for message in state["messages"][-6:])
        note = llm.invoke(
            [
                SystemMessage(content=REFLECTION_PROMPT),
                HumanMessage(content=f"最近工具结果：\n{transcript}"),
            ]
        ).content
        safe_print(note)
        return {"messages": state["messages"], "reflection": str(note)}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("reflection", reflection_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "reflection")
    graph.add_edge("reflection", "agent")

    result = graph.compile().invoke(
        {"messages": [HumanMessage(content=task)], "reflection": ""},
        config={"recursion_limit": 12},
    )

    safe_print("\n最终回答:")
    safe_print(result["messages"][-1].content)
    print_workspace("运行后的 demo workspace")


if __name__ == "__main__":
    main()
