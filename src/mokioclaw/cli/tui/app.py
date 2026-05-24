from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Literal

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, Header, Input, RichLog, Static

from mokioclaw.cli.event_summary import EventSummary, shorten, summarize_event
from mokioclaw.cli.tui.approval import ApprovalGate, ApprovalModal
from mokioclaw.cli.tui.logo import render_logo
from mokioclaw.core.approval import ApprovalDecision, ApprovalRequest
from mokioclaw.core.agent import stream_session_events
from mokioclaw.core.paths import default_workspace


StreamFactory = Callable[..., Iterable[dict[str, Any]]]


class AgentEventMessage(Message):
    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__()
        self.event = event


class RunFinishedMessage(Message):
    def __init__(self, status: str) -> None:
        super().__init__()
        self.status = status


class ApprovalRequestedMessage(Message):
    def __init__(self, gate: ApprovalGate) -> None:
        super().__init__()
        self.gate = gate


class MokioClawTuiApp(App[None]):
    CSS = """
    Screen {
        background: $surface;
    }

    #root {
        height: 1fr;
    }

    #top {
        height: 9;
        border-bottom: solid $primary;
        padding: 0 1;
    }

    #logo {
        width: 40;
        height: 8;
        content-align: center middle;
    }

    #title-block {
        width: 1fr;
        height: 8;
        padding-left: 1;
    }

    #title {
        text-style: bold;
        color: $primary;
    }

    #status {
        color: $text-muted;
    }

    #body {
        height: 1fr;
    }

    #events {
        width: 1fr;
        height: 100%;
        border-right: solid $panel;
    }

    #sidebar {
        width: 34;
        min-width: 28;
        height: 100%;
        padding: 1;
    }

    #side-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #input-row {
        height: 3;
        border: round $primary;
        padding: 0 1;
    }

    #prompt {
        width: 3;
        height: 1;
        content-align: center middle;
        color: $primary;
        text-style: bold;
    }

    #task-input {
        width: 1fr;
        height: 1;
        border: none;
        background: $surface;
    }

    #hint {
        color: $text-muted;
        width: 30;
        height: 1;
        padding-left: 1;
        content-align: right middle;
    }
    """

    BINDINGS = [
        ("ctrl+c", "cancel_or_quit", "Cancel/Quit"),
        ("ctrl+l", "clear_events", "Clear"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        initial_task: str | None = None,
        workspace: Path | None = None,
        max_attempts: int = 3,
        approval_mode: Literal["inline", "auto", "deny"] = "inline",
        checkpoint_mode: Literal["light", "strict", "off"] = "light",
        trace_mode: Literal["on", "off"] = "on",
        resume: Path | None = None,
        stream_factory: StreamFactory = stream_session_events,
    ) -> None:
        super().__init__()
        self.initial_task = initial_task
        self.workspace = resume or workspace or default_workspace()
        self.session_workspace = self.workspace
        self.max_attempts = max_attempts
        self.approval_mode = approval_mode
        self.checkpoint_mode = checkpoint_mode
        self.trace_mode = trace_mode
        self.resume = resume
        self.stream_factory = stream_factory
        self.running = False
        self.run_count = 0
        self.approval_count = 0
        self.failed_tool_count = 0
        self.tool_count = 0
        self.latest_workspace = str(self.session_workspace)
        self.latest_checkpoint = ""
        self.latest_trace = ""
        self.session_id = ""
        self.session_turn = 0
        self.last_route = ""
        self.sidebar_text = ""
        self.todos: list[dict[str, Any]] = []
        self._state_lock = Lock()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="root"):
            with Horizontal(id="top"):
                yield Static(render_logo(max_width=34, max_rows=8), id="logo")
                with Vertical(id="title-block"):
                    yield Static("MokioClaw TUI", id="title")
                    yield Static("ready", id="status")
                    yield Static("MultiAgent + Context/Harness Engineering", id="subtitle")
            with Horizontal(id="body"):
                yield RichLog(id="events", wrap=True, highlight=True, markup=True)
                with Vertical(id="sidebar"):
                    yield Static("Run State", id="side-title")
                    yield Static("", id="side-state")
            with Horizontal(id="input-row"):
                yield Static("❯", id="prompt")
                yield Input(placeholder="Chat or ask for coding work, then press Enter", id="task-input")
                yield Static("Enter send · /new session · Ctrl+L clear", id="hint")
        yield Footer()

    def on_mount(self) -> None:
        self._write_welcome()
        self._refresh_sidebar()
        self.query_one("#task-input", Input).focus()
        if self.initial_task:
            self.call_after_refresh(self.start_task, self.initial_task, self.resume)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "task-input":
            return
        task = event.value.strip()
        if not task or self.running:
            return
        event.input.value = ""
        if task == "/new":
            self.start_new_session()
            return
        self.start_task(task, None)

    def on_agent_event_message(self, message: AgentEventMessage) -> None:
        self._handle_event(message.event)

    def on_run_finished_message(self, message: RunFinishedMessage) -> None:
        self.running = False
        self.resume = None
        self.query_one("#task-input", Input).disabled = False
        self.query_one("#task-input", Input).focus()
        self.query_one("#status", Static).update(f"{message.status}; ready for next task")
        self._refresh_sidebar()

    def on_approval_requested_message(self, message: ApprovalRequestedMessage) -> None:
        workspace = self.latest_workspace or str(self.workspace or "")
        self.push_screen(ApprovalModal(message.gate.request, workspace), self._resolve_approval(message.gate))

    def action_cancel_or_quit(self) -> None:
        if self.running:
            self.notify("A run is active. Press Ctrl+Q to quit and let checkpoint handle recovery.", severity="warning")
            return
        self.exit()

    def action_clear_events(self) -> None:
        self.query_one("#events", RichLog).clear()
        self._write_welcome()

    def start_task(self, task: str, resume: Path | None = None) -> None:
        if self.running:
            self.notify("MokioClaw is already running a task.", severity="warning")
            return
        self.running = True
        self.run_count += 1
        self.todos = []
        self.failed_tool_count = 0
        self.tool_count = 0
        self.query_one("#task-input", Input).disabled = True
        self.query_one("#status", Static).update("running")
        self._refresh_sidebar()
        self._write_run_start(task, resume)
        self.run_worker(lambda: self._run_stream(task, resume), thread=True, exclusive=False, name=f"mokioclaw-run-{self.run_count}")

    def _run_stream(self, task: str, resume: Path | None) -> None:
        status = "finished"
        try:
            approval_handler = self._approval_handler if self.approval_mode == "inline" else None
            for event in self.stream_factory(
                task,
                session_workspace=self.session_workspace,
                max_attempts=self.max_attempts,
                approval_mode=self.approval_mode,
                approval_handler=approval_handler,
                checkpoint_mode=self.checkpoint_mode,
                resume_workspace=resume,
                trace_mode=self.trace_mode,
            ):
                self.call_from_thread(self.post_message, AgentEventMessage(event))
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception as exc:
            status = "failed"
            error_event = {"type": "custom_event", "event": {"type": "tui_error", "error": f"{type(exc).__name__}: {exc}"}}
            self.call_from_thread(self.post_message, AgentEventMessage(error_event))
        finally:
            self.call_from_thread(self.post_message, RunFinishedMessage(status))

    def _approval_handler(self, request: ApprovalRequest) -> ApprovalDecision:
        gate = ApprovalGate(request)
        self.call_from_thread(self.post_message, ApprovalRequestedMessage(gate))
        return gate.wait()

    def _resolve_approval(self, gate: ApprovalGate) -> Callable[[bool | None], None]:
        def resolve(result: bool | None) -> None:
            approved = bool(result)
            gate.resolve(approved)
            self._refresh_sidebar()

        return resolve

    def _handle_event(self, event: dict[str, Any]) -> None:
        self._update_state_from_event(event)
        summary = summarize_event(event)
        self._write_summary(summary)
        self._refresh_sidebar()

    def _update_state_from_event(self, event: dict[str, Any]) -> None:
        with self._state_lock:
            if event.get("type") == "workspace":
                self.latest_workspace = str(event.get("path", ""))
                self.session_workspace = Path(self.latest_workspace)
                return
            payload = event.get("event")
            if event.get("type") == "graph_event" and isinstance(payload, dict):
                for update in payload.values():
                    if isinstance(update, dict):
                        self._update_from_payload(update)
            elif event.get("type") == "custom_event" and isinstance(payload, dict):
                self._update_from_payload(payload)

    def _update_from_payload(self, payload: dict[str, Any]) -> None:
        if isinstance(payload.get("todos"), list):
            self.todos = payload["todos"]
        if payload.get("type") == "tool_call":
            self.tool_count += 1
        if payload.get("type") == "tool_result":
            result = payload.get("result")
            if isinstance(result, dict):
                if result.get("ok") is False:
                    self.failed_tool_count += 1
                if result.get("requires_approval"):
                    self.approval_count += 1
        if payload.get("type") == "checkpoint_saved":
            self.latest_checkpoint = str(payload.get("path", ""))
        if payload.get("type") == "trace_summary":
            self.latest_trace = str(payload.get("trace_dir", ""))
        if payload.get("type") == "session_started":
            self.session_id = str(payload.get("session_id", ""))
            self.session_turn = int(payload.get("turn_index", 0) or 0)
            self.latest_workspace = str(payload.get("workspace", self.latest_workspace))
        if payload.get("type") == "session_turn_started":
            self.session_turn = int(payload.get("turn", self.session_turn) or self.session_turn)
        if payload.get("type") == "session_turn_saved":
            self.session_turn = int(payload.get("turn", self.session_turn) or self.session_turn)
            self.last_route = str(payload.get("route", self.last_route))

    def _write_welcome(self) -> None:
        log = self.query_one("#events", RichLog)
        log.write(
            Panel(
                "Enter a message to start a persistent coding session. Use /new to open a fresh workspace.",
                title="MokioClaw",
                border_style="cyan",
            )
        )

    def _write_run_start(self, task: str, resume: Path | None) -> None:
        mode = f"resume: {resume}" if resume is not None else f"session workspace: {self.session_workspace}"
        self.query_one("#events", RichLog).write(
            Panel(shorten(task, 1000) + f"\n\n{mode}", title=f"Turn {self.run_count}", border_style="magenta")
        )

    def _write_summary(self, summary: EventSummary) -> None:
        self.query_one("#events", RichLog).write(
            Panel(summary.body or " ", title=summary.title, border_style=summary.style)
        )

    def _refresh_sidebar(self) -> None:
        status = "running" if self.running else "ready"
        workspace = shorten(self.latest_workspace or str(self.session_workspace), 80)
        checkpoint = shorten(self.latest_checkpoint or "(waiting)", 80)
        trace = shorten(self.latest_trace or "(waiting)", 80)
        tools = f"{self.tool_count} total / {self.failed_tool_count} failed"
        approvals = str(self.approval_count)
        todos = self._todo_sidebar_text()
        self.sidebar_text = "\n".join(
            [
                f"status {status}",
                f"turns {self.run_count}",
                f"session {self.session_id}",
                f"route {self.last_route or '(none)'}",
                f"workspace {workspace}",
                f"checkpoint {checkpoint}",
                f"trace {trace}",
                f"tools {tools}",
                f"approvals {approvals}",
                f"todos {todos}",
            ]
        )
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_row("status", status)
        table.add_row("turns", str(self.run_count))
        table.add_row("session", shorten(self.session_id or "(starting)", 24))
        table.add_row("route", self.last_route or "(none)")
        table.add_row("workspace", workspace)
        table.add_row("checkpoint", checkpoint)
        table.add_row("trace", trace)
        table.add_row("tools", tools)
        table.add_row("approvals", approvals)
        table.add_row("todos", todos)
        self.query_one("#side-state", Static).update(table)

    def _todo_sidebar_text(self) -> str:
        if not self.todos:
            return "(none yet)"
        counts: dict[str, int] = {}
        for todo in self.todos:
            status = str(todo.get("status", "pending"))
            counts[status] = counts.get(status, 0) + 1
        current = next((todo for todo in self.todos if todo.get("status") == "in_progress"), None)
        count_text = ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))
        if current:
            return f"{count_text}\n{shorten(current.get('content', current.get('description', '')), 120)}"
        return count_text

    def start_new_session(self) -> None:
        if self.running:
            self.notify("MokioClaw is already running a task.", severity="warning")
            return
        self.workspace = default_workspace()
        self.session_workspace = self.workspace
        self.resume = None
        self.latest_workspace = str(self.session_workspace)
        self.latest_checkpoint = ""
        self.latest_trace = ""
        self.session_id = ""
        self.session_turn = 0
        self.last_route = ""
        self.todos = []
        self.failed_tool_count = 0
        self.tool_count = 0
        self.approval_count = 0
        self._refresh_sidebar()
        self.query_one("#events", RichLog).write(
            Panel(str(self.session_workspace), title="New Session", border_style="cyan")
        )
