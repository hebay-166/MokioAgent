from __future__ import annotations

from demo_file_tools import TOOL_REGISTRY, list_files, reset_demo_workspace


def parse_model_output(model_output: str) -> tuple[str, dict[str, str]]:
    tool_name, payload = model_output.split(":", maxsplit=1)
    source, target = payload.split("->", maxsplit=1)
    return tool_name.strip(), {
        "source": source.strip(),
        "target": target.strip(),
    }


def main() -> None:
    reset_demo_workspace()

    print("=== 01. 最小 ToolCall：字符串协议 ===")
    print("\n运行前的 demo workspace:")
    print(list_files("."))

    model_output = "move_file:demo/inbox/a.txt->demo/archive/a.txt"
    print("\n假设模型输出:")
    print(model_output)

    tool_name, tool_args = parse_model_output(model_output)
    print("\n解析后的工具请求:")
    print(f"tool_name = {tool_name}")
    print(f"tool_args = {tool_args}")

    tool = TOOL_REGISTRY[tool_name]
    result = tool(**tool_args)
    print("\n工具执行结果:")
    print(result)

    print("\n运行后的 demo workspace:")
    print(list_files("."))


if __name__ == "__main__":
    main()
