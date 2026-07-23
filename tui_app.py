"""Textual front-end for parrotbox: session setup, a live conversation pane,
and real-time synthesis sliders.

The app owns no pipeline logic. It collects a SessionConfig on the setup
screen, hands it to the `start_session` callback supplied by main.py, then
shows the running conversation. Slider moves mutate the shared RuntimeControls
in place — the TTS service reads it on the next chunk, so changes are audible
within a sentence.
"""

from __future__ import annotations

import asyncio
import queue
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    RadioButton,
    RadioSet,
    RichLog,
    Static,
)

# Emitted (at DEBUG) by pipecat when the StartFrame reaches the end of the
# pipeline — i.e. every model is loaded and warm. We watch the log stream for
# it to flip the TUI from "booting" to "live".
READY_MARKER = "pipeline is now ready"

from runtime_controls import PARAM_SPECS, ParamSpec, RuntimeControls


class TranscriptBus:
    """One-way channel from the pipeline's frame taps to the TUI.

    put_nowait is called from the pipeline event loop (same loop as the app),
    so it never blocks; the app drains it in a background task.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    def user(self, text: str) -> None:
        self.queue.put_nowait(("user", text))

    def bot(self, text: str) -> None:
        self.queue.put_nowait(("bot", text))


@dataclass
class SessionSetup:
    scenario: str | None
    half_duplex: bool
    meeting: bool


@dataclass
class SessionHandles:
    """Callbacks the running pipeline hands back to the live screen."""

    respond: Callable[[], None]  # meeting mode: release one held reply ('r')
    speak: Callable[[str], None]  # say a typed line verbatim ('t')
    mute_toggle: Callable[[], bool]  # toggle mic mute ('m'); returns new muted state
    clear: Callable[[], None]  # reset conversation to a fresh context ('c')


# --------------------------------------------------------------------------- #
# Setup screen — the old interactive prompts, now on-screen.
# --------------------------------------------------------------------------- #


class SetupScreen(Screen):
    CSS = """
    SetupScreen { align: center middle; }
    #setup { width: 72; height: auto; padding: 1 2; border: round $primary; }
    #setup > .label { margin-top: 1; text-style: bold; }
    #setup Input { margin-top: 1; }
    #setup RadioSet { margin-top: 1; }
    #start { margin-top: 1; width: 100%; }
    #hint { color: $text-muted; margin-top: 1; }
    """

    def __init__(self, scenario_default: str | None) -> None:
        super().__init__()
        self._scenario_default = scenario_default

    def compose(self) -> ComposeResult:
        with Container(id="setup"):
            yield Static("parrotbox — session setup", classes="label")
            yield Static("Scenario (e.g. 'scrum standup') — optional", classes="label")
            yield Input(
                value=self._scenario_default or "",
                placeholder="leave blank to skip",
                id="scenario",
            )
            yield Static("Mic mode", classes="label")
            with RadioSet(id="mic"):
                yield RadioButton("Full duplex — needs headphones", value=True)
                yield RadioButton("Half duplex — open speakers (mutes mic while talking)")
            yield Static("Response mode", classes="label")
            with RadioSet(id="mode"):
                yield RadioButton("Live — reply every turn", value=True)
                yield RadioButton("Meeting — hold replies until you press 'r'")
            yield Button("Start session", variant="primary", id="start")
            yield Static(
                "Voice/emotion sliders are adjustable live once the session starts.",
                id="hint",
            )

    def on_mount(self) -> None:
        self.query_one("#scenario", Input).focus()

    def _submit(self) -> None:
        scenario = self.query_one("#scenario", Input).value.strip() or None
        half_duplex = self.query_one("#mic", RadioSet).pressed_index == 1
        meeting = self.query_one("#mode", RadioSet).pressed_index == 1
        self.app.begin_session(SessionSetup(scenario, half_duplex, meeting))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()


# --------------------------------------------------------------------------- #
# Live screen — conversation + sliders.
# --------------------------------------------------------------------------- #


class ConvLine(Static):
    """One turn in the conversation. Bot lines grow as chunks stream in."""

    ROLE_STYLE = {"you": "bold cyan", "bot": "bold magenta"}

    def __init__(self, role: str, text: str) -> None:
        super().__init__()
        self._role = role
        self._text = text
        self._redraw()

    def append(self, text: str) -> None:
        joiner = "" if not self._text or self._text.endswith((" ", "\n")) else " "
        self._text = f"{self._text}{joiner}{text}"
        self._redraw()

    def _redraw(self) -> None:
        style = self.ROLE_STYLE.get(self._role, "white")
        label = "you" if self._role == "you" else "bot"
        self.update(f"[{style}]{label} ›[/] {self._text}")


class ParamBar(Static):
    """A labelled slider bound to one RuntimeControls field.

    Every bar is the same length; importance is shown by *thickness* — the
    bar's glyph height (spec.emphasis), not its length. The filled fraction is
    the bright portion, the remainder is dimmed. An optional spec.note is shown
    after the value.
    """

    WIDTH = 20
    # Bottom-aligned block glyphs of increasing height => increasing thickness.
    THICKNESS = {"thin": "▂", "normal": "▄", "thick": "█"}

    def __init__(self, spec: ParamSpec, controls: RuntimeControls) -> None:
        super().__init__()
        self.spec = spec
        self.controls = controls
        self._selected = False

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.redraw()

    def redraw(self) -> None:
        val = getattr(self.controls, self.spec.attr)
        frac = (val - self.spec.lo) / (self.spec.hi - self.spec.lo)
        filled = round(frac * self.WIDTH)
        glyph = self.THICKNESS.get(self.spec.emphasis, "▄")
        marker = "▸" if self._selected else " "
        name_style = "bold" if self._selected else "dim"
        bar_style = "green" if self._selected else "green3"
        # One continuous bar of the emphasis glyph: bright = filled, dim = rest.
        bar = f"[{bar_style}]{glyph * filled}[/][dim]{glyph * (self.WIDTH - filled)}[/]"
        note = f" [b yellow]{self.spec.note}[/]" if self.spec.note else ""
        self.update(
            f"[{name_style}]{marker} {self.spec.label:<12}[/] "
            f"{bar} [b]{val:>4.2f}[/]{note}"
        )


class LiveScreen(Screen):
    # Don't auto-focus any widget on mount: otherwise Textual grabs the first
    # focusable one (the hidden type-box / boot-log pane) and it eats the arrow
    # keys until the user pressed Escape. Must be "" (not None) — a None here
    # falls back to the App's AUTO_FOCUS ("*"); "" disables it outright, so focus
    # stays on the screen and the sliders respond immediately.
    AUTO_FOCUS = ""

    CSS = """
    LiveScreen { layout: vertical; }
    #stage {
        height: 1fr; border: round $primary; padding: 0 1; margin: 0 1;
    }
    #bootlog { height: 1fr; }
    #conversation { height: 1fr; }
    .hidden { display: none; }
    #controls-panel {
        height: auto; border: round $secondary; padding: 0 1; margin: 0 1 1 1;
    }
    #controls-title { text-style: bold; color: $text-muted; }
    #speakbox { margin: 0 1; border: round $accent; }
    #status { height: 1; padding: 0 2; color: $text-muted; }
    ParamBar { height: 1; }
    ConvLine { height: auto; margin-bottom: 1; }
    """

    BINDINGS = [
        ("up", "select_prev", "Prev bar"),
        ("down", "select_next", "Next bar"),
        ("left", "decr", "–"),
        ("right", "incr", "+"),
        ("shift+left", "decr_big", ""),
        ("shift+right", "incr_big", ""),
        ("t", "type_message", "Type"),
        ("m", "mute", "Mute"),
        ("c", "clear", "Clear"),
        ("r", "respond", "Respond"),
        ("q", "quit", "Quit"),
        ("escape", "cancel_type", ""),
    ]

    def __init__(
        self,
        controls: RuntimeControls,
        bus: TranscriptBus,
        meeting: bool,
        on_respond: Callable[[], None],
        on_speak: Callable[[str], None],
        on_mute_toggle: Callable[[], bool],
        on_clear: Callable[[], None],
        log_queue: queue.Queue[str],
        on_ready: Callable[[], None],
    ) -> None:
        super().__init__()
        self._controls = controls
        self._bus = bus
        self._meeting = meeting
        self._on_respond = on_respond
        self._on_speak = on_speak
        self._on_mute_toggle = on_mute_toggle
        self._on_clear = on_clear
        self._log_queue = log_queue
        self._on_ready = on_ready
        self._bars: list[ParamBar] = []
        self._selected = 0
        self._current_bot: ConvLine | None = None
        self._drain_task: asyncio.Task | None = None
        self._log_timer = None
        self._ready = False  # readiness marker seen
        self._live = False  # boot log replaced by transcript
        self._muted = False
        self._clear_armed = False  # first 'c' press seen, waiting for confirm
        self._clear_timer = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="stage"):
            yield RichLog(id="bootlog", wrap=True, markup=True, auto_scroll=True, max_lines=1000)
            yield VerticalScroll(id="conversation", classes="hidden")
        with Container(id="controls-panel"):
            yield Static("Voice controls", id="controls-title")
            for spec in PARAM_SPECS:
                bar = ParamBar(spec, self._controls)
                self._bars.append(bar)
                yield bar
        yield Input(
            placeholder="Type a line for the voice to speak — Enter to send, Esc to cancel",
            id="speakbox",
            classes="hidden",
        )
        yield Static(self._status_text(), id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "parrotbox"
        self._refresh_subtitle()
        # The log/conversation panes are scrollable and would otherwise grab
        # focus and eat the arrow keys — keep focus on the screen so ←/→/↑/↓
        # always drive the sliders.
        self.query_one("#bootlog", RichLog).can_focus = False
        self.query_one("#conversation", VerticalScroll).can_focus = False
        self.set_focus(None)
        for i, bar in enumerate(self._bars):
            bar.set_selected(i == self._selected)
        self.query_one("#bootlog", RichLog).write("[dim]loading models — first run takes ~1 min…[/]")
        self._log_timer = self.set_interval(0.1, self._pump_logs)
        self._drain_task = asyncio.create_task(self._drain())

    def on_unmount(self) -> None:
        if self._drain_task:
            self._drain_task.cancel()

    # -- boot log -> ready transition ---------------------------------------- #

    def _pump_logs(self) -> None:
        log = self.query_one("#bootlog", RichLog)
        while True:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            line = line.rstrip("\n")
            if not self._live:
                log.write(line)
            if not self._ready and READY_MARKER in line:
                self._on_ready_detected()

    def _on_ready_detected(self) -> None:
        self._ready = True
        self._on_ready()  # detach the boot-log sink so debug spam stops
        log = self.query_one("#bootlog", RichLog)
        log.write("")
        log.write("[bold green]  ✓ Loaded![/]")
        self.set_timer(3.0, self._go_live)

    def _go_live(self) -> None:
        self._live = True
        if self._log_timer:
            self._log_timer.stop()
        self.query_one("#bootlog", RichLog).add_class("hidden")
        self.query_one("#conversation", VerticalScroll).remove_class("hidden")
        self._refresh_status()

    def _status_text(self) -> str:
        if not self._live:
            return "loading models — first run takes ~1 min…"
        spec = PARAM_SPECS[self._selected]
        return f"{spec.label}: {spec.help}"

    def _refresh_status(self) -> None:
        self.query_one("#status", Static).update(self._status_text())

    def _flash_status(self, message: str) -> None:
        """Show a transient message on the status line (restored by _refresh_status)."""
        self.query_one("#status", Static).update(message)

    def _refresh_subtitle(self) -> None:
        base = "meeting mode" if self._meeting else "live"
        self.sub_title = f"{base} · 🔇 mic muted" if self._muted else base

    # -- transcript streaming ------------------------------------------------ #

    async def _drain(self) -> None:
        convo = self.query_one("#conversation", VerticalScroll)
        while True:
            role, text = await self._bus.queue.get()
            if role == "user":
                self._current_bot = None
                await convo.mount(ConvLine("you", text))
            else:  # bot chunk
                if self._current_bot is None:
                    self._current_bot = ConvLine("bot", text)
                    await convo.mount(self._current_bot)
                else:
                    self._current_bot.append(text)
            convo.scroll_end(animate=False)

    # -- slider actions ------------------------------------------------------ #

    def _reselect(self, index: int) -> None:
        self._bars[self._selected].set_selected(False)
        self._selected = index % len(self._bars)
        self._bars[self._selected].set_selected(True)
        self._refresh_status()

    def action_select_prev(self) -> None:
        self._reselect(self._selected - 1)

    def action_select_next(self) -> None:
        self._reselect(self._selected + 1)

    def _adjust(self, steps: int) -> None:
        spec = PARAM_SPECS[self._selected]
        self._controls.adjust(spec, steps)
        self._bars[self._selected].redraw()

    def action_incr(self) -> None:
        self._adjust(1)

    def action_decr(self) -> None:
        self._adjust(-1)

    def action_incr_big(self) -> None:
        self._adjust(10)

    def action_decr_big(self) -> None:
        self._adjust(-10)

    def action_respond(self) -> None:
        if self._meeting:
            self._on_respond()

    # -- type to speak ------------------------------------------------------- #

    def action_type_message(self) -> None:
        # Only once the pipeline is live (the TTS is loaded); ignored during boot.
        if not self._live:
            return
        box = self.query_one("#speakbox", Input)
        box.remove_class("hidden")
        box.focus()  # takes the keyboard, so letters/arrows type instead of driving sliders

    def action_cancel_type(self) -> None:
        self._disarm_clear()  # Esc also backs out of a pending clear confirmation
        self._close_speakbox()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "speakbox":
            return
        text = event.value.strip()
        if text:
            self._on_speak(text)
        self._close_speakbox()

    def _close_speakbox(self) -> None:
        box = self.query_one("#speakbox", Input)
        box.value = ""
        box.add_class("hidden")
        # Hand focus back to the screen so ←/→/↑/↓ drive the sliders again.
        self.set_focus(None)

    # -- mic mute ------------------------------------------------------------ #

    def action_mute(self) -> None:
        if not self._live:
            return
        self._muted = self._on_mute_toggle()
        self._refresh_subtitle()
        self._flash_status("🔇 mic muted" if self._muted else "🎙 mic live")

    # -- clear context (double-press) ---------------------------------------- #

    def action_clear(self) -> None:
        if not self._live:
            return
        if not self._clear_armed:
            self._clear_armed = True
            self._flash_status("clear conversation? press c again to confirm (Esc cancels)")
            self._clear_timer = self.set_timer(3.0, self._disarm_clear)
            return
        # Second press: confirmed.
        self._disarm_clear()
        self._on_clear()
        convo = self.query_one("#conversation", VerticalScroll)
        convo.remove_children()
        self._current_bot = None
        self._flash_status("conversation cleared — fresh context")

    def _disarm_clear(self) -> None:
        if self._clear_timer is not None:
            self._clear_timer.stop()
            self._clear_timer = None
        if self._clear_armed:
            self._clear_armed = False
            self._refresh_status()

    def action_quit(self) -> None:
        # Screen-level bindings resolve actions against the screen, so a bare
        # "quit" never reaches App.action_quit — hand off explicitly.
        self.app.exit()


# --------------------------------------------------------------------------- #
# App shell.
# --------------------------------------------------------------------------- #


class ParrotboxApp(App):
    """Owns the two screens and bridges setup -> a running session."""

    TITLE = "parrotbox"

    def __init__(
        self,
        *,
        controls: RuntimeControls,
        bus: TranscriptBus,
        scenario_default: str | None,
        start_session: Callable[[SessionSetup], Awaitable[SessionHandles]],
        log_queue: queue.Queue[str],
        on_ready: Callable[[], None],
    ) -> None:
        super().__init__()
        self._controls = controls
        self._bus = bus
        self._scenario_default = scenario_default
        self._start_session = start_session
        self._log_queue = log_queue
        self._on_ready = on_ready

    def on_mount(self) -> None:
        self.push_screen(SetupScreen(self._scenario_default))

    def begin_session(self, setup: SessionSetup) -> None:
        # Fire-and-forget: the factory builds the pipeline and starts the
        # worker, returning a respond() callback for meeting mode.
        self.run_worker(self._launch(setup), exclusive=True)

    async def _launch(self, setup: SessionSetup) -> None:
        handles = await self._start_session(setup)
        await self.switch_screen(
            LiveScreen(
                self._controls,
                self._bus,
                setup.meeting,
                handles.respond,
                handles.speak,
                handles.mute_toggle,
                handles.clear,
                self._log_queue,
                self._on_ready,
            )
        )
