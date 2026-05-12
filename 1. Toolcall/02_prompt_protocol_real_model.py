from __future__ import annotations

import json
import re
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from demo_file_tools import TOOL_REGISTRY, list_files, reset_demo_workspace
from llm_config import load_llm_config

DEFAULT_USER_PROMPT = "请把 inbox 里的 a.txt 移动到 archive 目录"

SYSTEM_PROMPT = """
你是一个只负责生成工具调用请求的助手。
你不能直接移动文件，只能告诉程序应该调用哪个工具。

当前 demo workspace 中有这些工具：

1. list_files(path: str)
   查看目录中的文件。

2. read_file(path: str)
   读取文件内容。

3. move_file(source: str, target: str)
   移动文件。路径必须是相对于 demo workspace 的相对路径。

当用户需要移动文件时，你必须严格输出下面的格式，不要输出任何额外解释：

<Tool>move_file</Tool>
<Args>{"source":"inbox/a.txt","target":"archive/a.txt"}</Args>

如果用户不需要调用工具，再输出普通文本。
""".strip()


def parse_tool_call(text: str) -> dict[str, object] | None:
    tool_match = re.search(r"<Tool>(.*?)</Tool>", text, re.DOTALL)
    args_match = re.search(r"<Args>(.*?)</Args>", text, re.DOTALL)

    if not tool_match:
        return None

    tool_name = tool_match.group(1).strip()
    args: dict[str, object] = {}
    if args_match:
        args = json.loads(args_match.group(1).strip())

    return {"tool": tool_name, "args": args}


def main() -> None:
    user_prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_USER_PROMPT
    reset_demo_workspace()

    print("=== 02. 真实模型 + 手写工具协议 ===")
    print("\n用户请求:")
    print(user_prompt)

    print("\n运行前的 demo workspace:")
    print(list_files("."))

    try:
        llm_config = load_llm_config()
    except RuntimeError as exc:
        print("\n环境变量配置错误:")
        print(exc)
        return

    llm = ChatOpenAI(
        **llm_config,
        temperature=0,
    )

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

    model_text = str(response.content)
    print("\n模型原始输出:")
    print(model_text)

    call = parse_tool_call(model_text)
    if call is None:
        print("\n没有解析到工具调用，模型返回的是普通文本。")
        return

    tool_name = str(call["tool"])
    tool_args = call["args"]
    print("\n解析后的工具请求:")
    print(f"tool_name = {tool_name}")
    print(f"tool_args = {tool_args}")

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    if not isinstance(tool_args, dict):
        raise TypeError(f"Tool args must be a dict: {tool_args}")

    result = TOOL_REGISTRY[tool_name](**tool_args)
    print("\n工具执行结果:")
    print(result)

    print("\n运行后的 demo workspace:")
    print(list_files("."))


if __name__ == "__main__":
    main()
