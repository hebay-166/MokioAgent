from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from multi_agent_common import (
    DEFAULT_TASK,
    list_files,
    load_llm_config,
    move_file,
    print_workspace,
    reset_demo_workspace,
    safe_print,
    write_file,
)


COORDINATOR_PROMPT = """
你是 coordinator expert。
你不直接操作文件，也不直接写代码，只负责给 expert 写交付单。
当前有两个 expert：
1. file_expert：负责移动、查看文件。
2. code_expert：负责生成简单 Python 代码。

请严格输出两行：
file_handoff: 一句话说明交给 file_expert 的任务
code_handoff: 一句话说明交给 code_expert 的任务
""".strip()

SUMMARY_PROMPT = """
你是 coordinator expert。
根据 file_expert 的执行结果，用中文总结最终目录变化。
""".strip()


def parse_handoff(text: str) -> tuple[str, str]:
    file_handoff = ""
    code_handoff = ""
    for line in text.splitlines():
        if line.startswith("file_handoff:"):
            file_handoff = line.split(":", maxsplit=1)[1].strip()
        if line.startswith("code_handoff:"):
            code_handoff = line.split(":", maxsplit=1)[1].strip()
    return file_handoff, code_handoff


def file_expert(handoff: str) -> str:
    safe_print("\n[file_expert] 接收 coordinator 交付:")
    safe_print(handoff)

    before = list_files(".")
    result = move_file("inbox/a.txt", "archive/a.txt")
    after = list_files(".")

    report = f"before:\n{before}\n\naction:\n{result}\n\nafter:\n{after}"
    safe_print("\n[file_expert] 执行报告:")
    safe_print(report)
    return report


def code_expert(handoff: str, file_report: str) -> str:
    safe_print("\n[code_expert] 接收 coordinator 交付:")
    safe_print(handoff)

    code = '''from pathlib import Path


def main():
    archive_file = Path("archive/a.txt")
    if archive_file.exists():
        print("整理完成：archive/a.txt 已存在")
    else:
        print("整理未完成：archive/a.txt 不存在")


if __name__ == "__main__":
    main()
'''
    result = write_file("check_archive.py", code)
    report = f"{result}\n\nbased_on:\n{file_report}"

    safe_print("\n[code_expert] 执行报告:")
    safe_print(report)
    return report


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_demo_workspace()

    safe_print("=== 01. MultiAgent：Coordinator -> File Expert + Code Expert ===")
    safe_print("\n用户任务:")
    safe_print(task)
    print_workspace("运行前的 demo workspace")

    llm = ChatOpenAI(**load_llm_config(), temperature=0)

    safe_print("\n[coordinator] 生成交付单")
    handoff_text = llm.invoke(
        [SystemMessage(content=COORDINATOR_PROMPT), HumanMessage(content=task)]
    ).content
    handoff_text = str(handoff_text).strip()
    safe_print(handoff_text)

    file_handoff, code_handoff = parse_handoff(handoff_text)

    if not file_handoff or not code_handoff:
        safe_print("\ncoordinator 交付单格式不完整，任务停止。")
        return

    file_report = file_expert(file_handoff)
    code_report = code_expert(code_handoff, file_report)

    safe_print("\n[coordinator] 汇总 expert 结果")
    final_report = f"file_expert:\n{file_report}\n\ncode_expert:\n{code_report}"
    summary = llm.invoke(
        [SystemMessage(content=SUMMARY_PROMPT), HumanMessage(content=final_report)]
    ).content
    safe_print(summary)

    print_workspace("运行后的 demo workspace")


if __name__ == "__main__":
    main()
