from __future__ import annotations

import shutil
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent / "demo_workspace"


def reset_demo_workspace() -> Path:
    """Reset the tiny workspace used by every ToolCall demo."""
    workspace = WORKSPACE.resolve()
    demo_root = Path(__file__).resolve().parent

    if workspace.exists():
        if workspace.parent != demo_root:
            raise RuntimeError(f"Refuse to reset unexpected path: {workspace}")
        shutil.rmtree(workspace, ignore_errors=True)

    (workspace / "inbox").mkdir(parents=True, exist_ok=True)
    (workspace / "archive").mkdir(parents=True, exist_ok=True)
    (workspace / "inbox" / "a.txt").write_text(
        "Hello from MokioClaw ToolCall demo.\n",
        encoding="utf-8",
    )
    return workspace


def _normalize_demo_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    for prefix in ("demo/", "demo_workspace/"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def resolve_workspace_path(path: str) -> Path:
    workspace = WORKSPACE.resolve()
    normalized = _normalize_demo_path(path)
    raw_path = Path(normalized)

    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        resolved = (workspace / raw_path).resolve()

    if not resolved.is_relative_to(workspace):
        raise ValueError(f"Path is outside demo workspace: {path}")

    return resolved


def display_path(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE.resolve())).replace("\\", "/")


def list_files(path: str = ".") -> str:
    target = resolve_workspace_path(path)
    if not target.exists():
        return f"{path} does not exist"

    if target.is_file():
        return display_path(target)

    files = sorted(item for item in target.rglob("*") if item.is_file())
    if not files:
        return "(empty)"

    return "\n".join(f"- {display_path(item)}" for item in files)


def read_file(path: str) -> str:
    target = resolve_workspace_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return target.read_text(encoding="utf-8")


def move_file(source: str, target: str) -> str:
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


TOOL_REGISTRY = {
    "list_files": list_files,
    "read_file": read_file,
    "move_file": move_file,
}
