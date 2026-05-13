from __future__ import annotations


def get_weather(city: str) -> str:
    weather = {
        "Beijing": "晴天，25 度",
        "Shanghai": "多云，28 度",
        "Hangzhou": "小雨，22 度",
    }
    return f"{city} 的天气是：{weather.get(city, '未知天气')}"


def parse_model_output(text: str) -> tuple[str, dict[str, str]]:
    tool_name, city = text.split(":", maxsplit=1)
    return tool_name.strip(), {"city": city.strip()}


def main() -> None:
    print("=== 01. 字符串协议版 ToolCall ===")

    model_output = "get_weather:Beijing"

    print("\n假设模型输出:")
    print(model_output)

    tool_name, tool_args = parse_model_output(model_output)
    
    print("\n解析后的工具请求:")
    print(f"tool_name = {tool_name}")
    print(f"tool_args = {tool_args}")

    if tool_name == "get_weather":
        result = get_weather(**tool_args)
    else:
        result = f"未知工具：{tool_name}"

    print("\n工具执行结果:")
    print(result)


if __name__ == "__main__":
    main()
