from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv


WORKSPACE = Path(__file__).resolve().parent / "demo_workspace"
DEFAULT_TASK = "请把 inbox 里的 a.txt 移动到 archive，然后生成一份简单的 Python 代码。"


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
        "Hello from MokioClaw MultiAgent demo.\n",
        encoding="utf-8",
    )


def resolve_workspace_path(path: str) -> Path:
    workspace = WORKSPACE.resolve()
    raw_path = Path(path.strip().replace("\\", "/"))
    resolved = raw_path.resolve() if raw_path.is_absolute() else (workspace / raw_path).resolve()

    if not resolved.is_relative_to(workspace):
        raise ValueError(f"Path is outside demo workspace: {path}")

    return resolved


def display_path(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE.resolve())).replace("\\", "/")


def list_files(path: str = ".") -> str:
    target = resolve_workspace_path(path)
    files = sorted(item for item in target.rglob("*") if item.is_file())
    return "\n".join(f"- {display_path(item)}" for item in files) if files else "(empty)"


def move_file(source: str, target: str) -> str:
    source_path = resolve_workspace_path(source)
    target_path = resolve_workspace_path(target)

    if target_path.exists() and target_path.is_dir():
        target_path = target_path / source_path.name

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(target_path))
    return f"moved {display_path(target_path)}"


def write_file(path: str, content: str) -> str:
    target_path = resolve_workspace_path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    return f"created {display_path(target_path)}"


def safe_print(value: object = "") -> None:
    text = str(value)
    print(text.encode("gbk", errors="replace").decode("gbk"))


def print_workspace(title: str) -> None:
    safe_print(f"\n{title}:")
    safe_print(list_files("."))
