from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import demo_file_tools
from demo_file_tools import list_files as _list_files
from demo_file_tools import move_file as _move_file
from demo_file_tools import read_file as _read_file
from demo_file_tools import reset_demo_workspace
from llm_config import load_llm_config

DEFAULT_USER_PROMPT = "请把 inbox 里的 a.txt 移动到 archive 目录"

SYSTEM_PROMPT = """
你正在控制一个安全的 demo workspace。
如果用户要查看文件、读取文件或移动文件，请调用可用工具，不要直接声称已经完成。
所有路径都必须使用相对于 demo workspace 的相对路径，例如 inbox/a.txt 或 archive/a.txt。
如果用户明确要求移动文件，请直接调用 move_file，不要先调用 list_files 或 read_file。
本示例只演示一次 ToolCall，不需要做多轮总结。
""".strip()


@tool
def list_files(path: str = ".") -> str:
    """List files under a path inside the demo workspace."""
    return _list_files(path)


@tool
def read_file(path: str) -> str:
    """Read a file inside the demo workspace."""
    return _read_file(path)


@tool
def move_file(source: str, target: str) -> str:
    """Move one file inside the demo workspace."""
    return _move_file(source, target)


def main() -> None:
    user_prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_USER_PROMPT
    reset_demo_workspace()

    print("=== 03. LangChain 原生 ToolCall ===")
    print("\n用户请求:")
    print(user_prompt)

    print("\n运行前的 demo workspace:")
    print(demo_file_tools.list_files("."))

    tools = [list_files, read_file, move_file]
    tool_registry = {tool_item.name: tool_item for tool_item in tools}

    try:
        llm_config = load_llm_config()
    except RuntimeError as exc:
        print("\n环境变量配置错误:")
        print(exc)
        return

    llm = ChatOpenAI(
        **llm_config,
        temperature=0,
    ).bind_tools(tools)

    try:
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as exc:
        print("\n模型调用失败:")
        print(f"{type(exc).__name__}: {exc}")
        print(
            f"\n请确认模型 {llm_config['model']} 可以通过 OpenAI-compatible 接口访问。"
        )
        return

    print("\n模型文本输出:")
    print(response.content)

    print("\nLangChain 解析出的 tool_calls:")
    print(response.tool_calls)

    if not response.tool_calls:
        print("\n模型这次没有触发原生 tool call。")
        print("这通常说明当前模型或 OpenAI-compatible provider 对 native tool calling 支持不完整。")
        return

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print("\n准备执行工具:")
        print(f"tool_name = {tool_name}")
        print(f"tool_args = {tool_args}")

        result = tool_registry[tool_name].invoke(tool_args)
        print("\n工具执行结果:")
        print(result)

    print("\n运行后的 demo workspace:")
    print(demo_file_tools.list_files("."))
    print("\n单次 ToolCall 到这里结束；下一阶段 AgentLoop 会把工具结果再回填给模型。")


if __name__ == "__main__":
    main()
