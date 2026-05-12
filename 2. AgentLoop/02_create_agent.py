from __future__ import annotations

import sys

from langchain_openai import ChatOpenAI

from agent_loop_common import (
    DEFAULT_TASK,
    TOOLS,
    load_llm_config,
    print_workspace,
    reset_demo_workspace,
    safe_print,
)


SYSTEM_PROMPT = """
你是一个文件整理助手。
你可以使用工具循环完成任务。
先检查 inbox，再移动 a.txt 到 archive，最后查看整理后的目录并总结。
""".strip()


def main() -> None:
    try:
        from langchain.agents import create_agent
    except ModuleNotFoundError:
        safe_print("当前 Python 环境缺少 langchain，先运行 uv sync 或安装 pyproject.toml 中的依赖。")
        return

    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_demo_workspace()

    safe_print("=== 02. LangChain create_agent ===")
    safe_print("\n用户任务:")
    safe_print(task)
    print_workspace("运行前的 demo workspace")

    agent = create_agent(
        model=ChatOpenAI(**load_llm_config(), temperature=0),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    result = agent.invoke({"messages": [{"role": "user", "content": task}]})

    safe_print("\nAgent 消息流:")
    for message in result["messages"]:
        message_type = getattr(message, "type", "unknown")
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)
        safe_print(f"\n[{message_type}]")
        if tool_calls:
            safe_print(tool_calls)
        elif content:
            safe_print(content)

    print_workspace("运行后的 demo workspace")


if __name__ == "__main__":
    main()
