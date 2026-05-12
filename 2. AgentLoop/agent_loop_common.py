from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool


WORKSPACE = Path(__file__).resolve().parent / "demo_workspace"
DEFAULT_TASK = "请检查 inbox，把 a.txt 移动到 archive，然后告诉我整理后的目录变化。"


def load_llm_config() -> dict[str, str]:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)

    required_keys = ["MODEL", "BASE_URL", "API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        raise RuntimeError(f"Missing .env values: {', '.join(missing_keys)}")

    return {
        "model": os.environ["MODEL"],
        "base_url": os.environ["BASE_URL"],
        "api_key": os.environ["API_KEY"],
    }


def reset_demo_workspace() -> None:
    workspace = WORKSPACE.resolve()
    demo_root = Path(__file__).resolve().parent

    if workspace.exists():
        if workspace.parent != demo_root:
            raise RuntimeError(f"Refuse to reset unexpected path: {workspace}")
        shutil.rmtree(workspace, ignore_errors=True)

    (workspace / "inbox").mkdir(parents=True, exist_ok=True)
    (workspace / "archive").mkdir(parents=True, exist_ok=True)
    (workspace / "inbox" / "a.txt").write_text(
        "Hello from MokioClaw AgentLoop demo.\n",
        encoding="utf-8",
    )


def _normalize_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    for prefix in ("demo/", "demo_workspace/"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def resolve_workspace_path(path: str) -> Path:
    workspace = WORKSPACE.resolve()
    raw_path = Path(_normalize_path(path))
    resolved = raw_path.resolve() if raw_path.is_absolute() else (workspace / raw_path).resolve()

    if not resolved.is_relative_to(workspace):
        raise ValueError(f"Path is outside demo workspace: {path}")

    return resolved


def display_path(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE.resolve())).replace("\\", "/")


def _list_files(path: str = ".") -> str:
    target = resolve_workspace_path(path)
    if not target.exists():
        return f"{path} does not exist"
    if target.is_file():
        return display_path(target)

    files = sorted(item for item in target.rglob("*") if item.is_file())
    return "\n".join(f"- {display_path(item)}" for item in files) if files else "(empty)"


def _read_file(path: str) -> str:
    target = resolve_workspace_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return target.read_text(encoding="utf-8")


def _move_file(source: str, target: str) -> str:
    source_path = resolve_workspace_path(source)
    target_path = resolve_workspace_path(target)

    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")

    target_text = target.strip().replace("\\", "/")
    if target_path.exists() and target_path.is_dir():
        target_path = target_path / source_path.name
    elif target_text.endswith("/"):
        target_path = target_path / source_path.name

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(target_path))
    return f"moved {display_path(target_path)}"


@tool
def list_files(path: str = ".") -> str:
    """List files under a path inside the demo workspace."""
    return _list_files(path)


@tool
def read_file(path: str) -> str:
    """Read one file inside the demo workspace."""
    return _read_file(path)


@tool
def move_file(source: str, target: str) -> str:
    """Move one file inside the demo workspace."""
    return _move_file(source, target)


TOOLS = [list_files, read_file, move_file]
TOOL_REGISTRY = {item.name: item for item in TOOLS}


def safe_print(value: object = "") -> None:
    text = str(value)
    print(text.encode("gbk", errors="replace").decode("gbk"))


def print_workspace(title: str) -> None:
    safe_print(f"\n{title}:")
    safe_print(_list_files("."))


def print_tool_call(tool_call: dict) -> None:
    safe_print(f"tool_name = {tool_call['name']}")
    safe_print(f"tool_args = {tool_call['args']}")
