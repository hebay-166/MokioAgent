from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

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


SYSTEM_PROMPT = """
你是一个文件整理助手。
你可以反复调用工具，直到完成用户任务。
推荐顺序：先 list_files("inbox")，再 move_file("inbox/a.txt", "archive/a.txt")，再 list_files(".")，最后总结。
每一轮最多调用一个工具。
""".strip()


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_demo_workspace()

    safe_print("=== 01. 手写 while AgentLoop ===")
    safe_print("\n用户任务:")
    safe_print(task)
    print_workspace("运行前的 demo workspace")

    llm = ChatOpenAI(**load_llm_config(), temperature=0).bind_tools(TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=task)]

    for turn in range(1, 8):
        safe_print(f"\n--- 第 {turn} 轮：模型思考 ---")
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            safe_print("\n最终回答:")
            safe_print(response.content)
            break

        for tool_call in response.tool_calls:
            safe_print("\n模型决定调用工具:")
            print_tool_call(tool_call)

            result = TOOL_REGISTRY[tool_call["name"]].invoke(tool_call["args"])
            safe_print("\n工具返回:")
            safe_print(result)

            messages.append(
                ToolMessage(
                    content=str(result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
    else:
        safe_print("\n达到最大轮数，AgentLoop 停止。")

    print_workspace("运行后的 demo workspace")


if __name__ == "__main__":
    main()
