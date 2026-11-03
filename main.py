"""Apple Silicon voice agent: local STT → LLM endpoint → local cloned-voice TTS.

See ARCHITECTURE.md for the full design. The Textual interface is the default:
on-screen setup, a live conversation pane, and sliders for
exaggeration/temperature/cfg_weight/speed/volume that take effect mid-session.
Logs go to logs/ instead of the console in that mode.
    DEEPSEEK_API_KEY=... uv run python main.py

Or via OpenRouter (any of their models; see llm_provider.py):

    OPENROUTER_API_KEY=... uv run python main.py

Add --console for the classic prompts + log-stream flow:

    DEEPSEEK_API_KEY=... uv run python main.py --console
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Chatterbox runs tqdm progress bars inside every generate() call; in a
# long-running agent that's pure log spam. Must be set before chatterbox loads.
os.environ.setdefault("TQDM_DISABLE", "1")

from loguru import logger

# Only cheap imports at module level. Pipecat and everything that pulls it in
# (smart-turn → torch, VAD, transports) costs ~3.5s of import time; it is all
# imported lazily inside the functions below so the TUI's setup screen renders
# immediately instead of after a multi-second blank.
from clone_voice import reference_path
from llm_provider import (
    llm_api_key,
    llm_base_url,
    llm_config_errors,
    llm_model,
    resolve_llm_provider,
)
from persona import build_system_prompt
from runtime_controls import RuntimeControls

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext

    from meeting_mode import MeetingGate
    from mic_gate import MicGate

# 1.0 is natural rate. Speeds above 1.0 play faster but can sound echoey.
DEFAULT_SPEED = 1.0


@dataclass
class SessionConfig:
    scenario: str | None = None
    speed: float = DEFAULT_SPEED
    half_duplex: bool = False
    meeting: bool = False


@dataclass
class BuiltPipeline:
    """What build_pipeline hands back: the worker plus the live-control hooks."""

    worker: PipelineWorker
    meeting_gate: MeetingGate | None
    mic_gate: MicGate
    context: "LLMContext"
    system_content: str  # the original system prompt, for clearing back to fresh


def _device_index(env_name: str) -> int | None:
    value = os.environ.get(env_name)
    return int(value) if value else None


def _scenario_from_argv() -> str | None:
    """Command-line scenario words, ignoring flags like --console."""
    words = [a for a in sys.argv[1:] if not a.startswith("-")]
    return " ".join(words) or None


def get_session_config() -> SessionConfig:
    """Interactive session setup, asked before any model loads.

    A scenario given on the command line skips that question. Enter accepts
    each default. Non-interactive runs (no tty) get the defaults.
    """
    config = SessionConfig()
    scenario = _scenario_from_argv()
    if scenario:
        config.scenario = scenario
    if not sys.stdin.isatty():
        return config

    if config.scenario is None:
        answer = input("Scenario for this session, e.g. 'scrum standup' (enter to skip): ")
        config.scenario = answer.strip() or None

    while True:
        answer = input(f"Voice speed [{DEFAULT_SPEED}=natural]; above 1.0 is faster but can sound echoey: ").strip()
        if not answer:
            break
        try:
            config.speed = float(answer)
            if 0.5 <= config.speed <= 2.0:
                break
            print("keep it between 0.5 and 2.0")
        except ValueError:
            print("not a number")

    answer = input("Mic mode — (f)ull duplex needs headphones, (h)alf for open speakers [f]: ")
    config.half_duplex = answer.strip().lower().startswith("h")

    answer = input("Response mode — (l)ive replies every turn, (m)eeting: hold until you press 'r' [l]: ")
    config.meeting = answer.strip().lower().startswith("m")
    if config.meeting and config.half_duplex:
        print(
            "note: half duplex mutes the mic while the agent talks, "
            "so meeting speech during replies is lost"
        )
    return config


def build_pipeline(
    config: SessionConfig | None = None,
    *,
    controls: RuntimeControls | None = None,
    bus=None,
    metrics_jsonl: Path | None = None,
) -> BuiltPipeline:
    # Heavy imports live here, not at module top: they cost ~3.5s and aren't
    # needed until a session actually starts (see _warm_imports).
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )
    from pipecat.turns.user_mute import AlwaysUserMuteStrategy
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    from meeting_mode import MeetingGate
    from mic_gate import MicGate
    from stt_parakeet import ParakeetMLXSTTService
    from transcript_tap import TranscriptTap
    from tts_chatterbox import ChatterboxTTSService
    from tts_normalize import TTSNormalizeFilter
    from turn_metrics import TTSChunkLog, TurnMetricsObserver

    config = config or SessionConfig()
    # Seed the live controls from the session's starting speed; the TUI mutates
    # this same object in place while the pipeline runs.
    controls = controls or RuntimeControls(speed=config.speed)
    stt = ParakeetMLXSTTService()
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,  # Silero + Parakeet native rate
            audio_out_sample_rate=24000,  # Chatterbox native rate
            # Defaults (None) use the system default devices. Set to route
            # elsewhere, e.g. a BlackHole virtual device for app audio —
            # enumerate with scripts/list_audio_devices.py.
            input_device_index=_device_index("AUDIO_IN_DEVICE"),
            output_device_index=_device_index("AUDIO_OUT_DEVICE"),
        )
    )

    provider = resolve_llm_provider()
    llm = OpenAILLMService(
        api_key=llm_api_key(provider),
        base_url=llm_base_url(provider),
        settings=OpenAILLMService.Settings(
            model=llm_model(provider),
            extra={"extra_body": provider.extra_body},
        ),
    )

    # Per-chunk synthesis stats live here; the observer drains them at the end
    # of each turn to build one stitched latency line.
    chunk_log = TTSChunkLog()

    tts = ChatterboxTTSService(
        # Read at call time, so a voice cloned earlier in this same TUI
        # process is picked up without a restart.
        reference_wav=reference_path(),
        controls=controls,
        chunk_log=chunk_log,
        # Rewrites LLM output into speakable form ("..." would be spoken as
        # "dot dot dot") — filtered text is also what history records.
        text_filters=[TTSNormalizeFilter()],
    )

    # Half-duplex for open speakers: LocalAudioTransport has no echo
    # cancellation, so without headphones the mic hears the speakers and the
    # agent barge-ins on itself, then converses with its own echo. Half duplex
    # mutes the mic while the bot talks — trade: no barge-in.
    mute_strategies = [AlwaysUserMuteStrategy()] if config.half_duplex else []

    system_content = build_system_prompt(config.scenario, meeting=config.meeting)
    context = LLMContext(messages=[{"role": "system", "content": system_content}])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.2, min_volume=0.6)
            ),
            user_mute_strategies=mute_strategies,
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(
                            params=SmartTurnParams(
                                stop_secs=2.5,  # fallback silence when model says "incomplete"
                                pre_speech_ms=500,
                                max_duration_secs=8.0,
                            )
                        )
                    )
                ]
            ),
        ),
    )

    # Meeting mode: the gate swallows the run trigger after each user turn
    # (context still accumulates upstream of it); pressing 'r' releases one run.
    gate = MeetingGate() if config.meeting else None

    # Optional transcript taps for the TUI: user text right after STT, bot text
    # right after TTS. Pass-through processors, so they don't alter the flow.
    user_tap = TranscriptTap(on_user=bus.user) if bus else None
    bot_tap = TranscriptTap(on_bot=bus.bot) if bus else None

    # Manual mic mute (TUI 'm'): drops input audio when muted. Right after the
    # transport so STT and VAD go silent while muted. Unmuted by default.
    mic_gate = MicGate()

    stages = [
        transport.input(),
        mic_gate,
        stt,
        *([user_tap] if user_tap else []),
        aggregators.user(),
        *([gate] if gate else []),
        llm,
        tts,
        *([bot_tap] if bot_tap else []),
        transport.output(),
        # After output() on purpose: history records only text whose audio
        # actually played, so barge-in leaves an honest context behind.
        aggregators.assistant(),
    ]

    worker = PipelineWorker(
        Pipeline(stages),
        params=PipelineParams(enable_metrics=True),
        # An observer sees every frame without sitting in the pipeline, so the
        # latency line costs nothing on the critical path (pipecat feeds each
        # observer from its own queue).
        observers=[TurnMetricsObserver(chunk_log, jsonl_path=metrics_jsonl)],
        idle_timeout_secs=None,  # personal always-on session; never self-cancel
    )
    return BuiltPipeline(
        worker=worker,
        meeting_gate=gate,
        mic_gate=mic_gate,
        context=context,
        system_content=system_content,
    )


def _check_prerequisites(*, console: bool):
    try:
        provider = resolve_llm_provider()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    errors = llm_config_errors(provider)
    if errors:
        logger.error(f"Invalid LLM configuration ({provider.name}): {'; '.join(errors)}")
        sys.exit(1)
    # The TUI has an in-app clone flow, so a missing reference is a setup-screen
    # state there, not a fatal error. The console flow has no way to fix it, so
    # it still fails fast.
    if console and not reference_path().exists():
        logger.error(
            f"Voice reference missing: {reference_path()} — "
            "run without --console and use 'Clone a voice' (see README.md)"
        )
        sys.exit(1)


def _make_respond(gate: MeetingGate | None, worker: PipelineWorker):
    """A press-'r' handler for meeting mode: release one held reply."""

    def on_r():
        from pipecat.frames.frames import LLMRunFrame

        if gate is None:
            return
        if gate.held == 0:
            logger.info("meeting mode: nothing new since last reply")
            return
        gate.arm()
        asyncio.ensure_future(worker.queue_frame(LLMRunFrame()))

    return on_r


def _make_speak(worker: PipelineWorker):
    """A handler for the TUI's 'type to speak' (T): say a typed line verbatim.

    A TTSSpeakFrame is synthesized directly by the TTS (bypassing the LLM) and,
    because pipecat marks it append_to_context, the assistant aggregator records
    it — so within the session the bot knows it said it, and it shows in the
    conversation pane like any spoken line.
    """

    def speak(text: str):
        from pipecat.frames.frames import TTSSpeakFrame

        text = text.strip()
        if not text:
            return
        asyncio.ensure_future(worker.queue_frame(TTSSpeakFrame(text)))

    return speak


def _make_mute_toggle(mic_gate: MicGate):
    """A toggle for the TUI 'm' key; returns the new muted state to display."""

    def toggle() -> bool:
        muted = mic_gate.toggle()
        logger.info("mic {}", "muted" if muted else "unmuted")
        return muted

    return toggle


def _make_clear(context: "LLMContext", system_content: str):
    """Reset the conversation to a fresh context (system prompt only) for the TUI 'c' key.

    Runs on the pipeline's own event loop (same loop as the TUI), so replacing
    the shared context's messages doesn't race the aggregators.
    """

    def clear():
        context.set_messages([{"role": "system", "content": system_content}])
        logger.info("conversation context cleared")

    return clear


async def run_cli():
    from datetime import datetime

    config = get_session_config()

    from pipecat.workers.runner import WorkerRunner

    from meeting_mode import install_key_listener
    logger.info(
        f"Session: scenario={config.scenario or '(none)'} speed={config.speed} "
        f"mode={'half' if config.half_duplex else 'full'}-duplex"
    )

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    metrics_jsonl = logs_dir / f"turns-{datetime.now():%Y%m%d-%H%M%S}.jsonl"

    built = build_pipeline(config, metrics_jsonl=metrics_jsonl)
    worker, gate = built.worker, built.meeting_gate
    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(worker)

    restore = None
    if gate and sys.stdin.isatty():
        restore = install_key_listener(asyncio.get_running_loop(), _make_respond(gate, worker))
        logger.info("Meeting mode: agent holds replies — press 'r' to respond")

    logger.info("Starting voice agent — models load on first frame, give it ~1 min")
    try:
        await runner.run()
    finally:
        if restore:
            restore()


def _warm_imports() -> None:
    """Pay the ~3.5s pipecat/torch import cost in a background thread while the
    user is on the setup screen, so clicking Start doesn't stall on it."""
    try:
        import pipecat.audio.turn.smart_turn.local_smart_turn_v3  # noqa: F401
        import pipecat.services.openai.llm  # noqa: F401
        import pipecat.transports.local.audio  # noqa: F401
        import pipecat.workers.runner  # noqa: F401

        import stt_parakeet  # noqa: F401
        import tts_chatterbox  # noqa: F401
    except Exception:
        logger.exception("background import warm-up failed (retried on session start)")


async def run_tui():
    import queue
    from datetime import datetime

    from tui_app import ParrotboxApp, SessionHandles, SessionSetup, TranscriptBus

    # The TUI owns the screen, so logs never go to stderr. Two sinks: a
    # persistent file, and a DEBUG queue the boot screen tails during startup
    # (so the loading logs land in the conversation box) until the pipeline is
    # ready, at which point on_ready() detaches it.
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    logfile = logs_dir / f"parrotbox-{stamp}.log"
    # One JSON object per turn, for offline analysis (RTF drift, gap trends).
    metrics_jsonl = logs_dir / f"turns-{stamp}.jsonl"
    logger.remove()
    logger.add(str(logfile), level="INFO")

    log_queue: queue.Queue[str] = queue.Queue()
    boot_sink_id = logger.add(
        lambda msg: log_queue.put(str(msg)),
        level="DEBUG",
        format="{time:HH:mm:ss} | {level: <7} | {message}",
    )

    def on_ready():
        try:
            logger.remove(boot_sink_id)
        except ValueError:
            pass  # already removed

    logger.info("TUI session starting; logs go here instead of the console")

    controls = RuntimeControls()
    bus = TranscriptBus()
    runners: list = []
    run_tasks: list[asyncio.Task] = []

    # Pay the pipecat/torch import cost in the background while the user is on
    # the setup screen, so Start doesn't stall on it.
    _warm_task = asyncio.create_task(asyncio.to_thread(_warm_imports))

    async def start_session(setup: SessionSetup):
        from pipecat.workers.runner import WorkerRunner

        config = SessionConfig(
            scenario=setup.scenario,
            half_duplex=setup.half_duplex,
            meeting=setup.meeting,
        )
        logger.info(f"Session: {config}")
        built = build_pipeline(
            config, controls=controls, bus=bus, metrics_jsonl=metrics_jsonl
        )
        worker = built.worker
        runner = WorkerRunner(handle_sigint=False)  # Textual owns the keyboard
        await runner.add_workers(worker)
        runners.append(runner)
        run_tasks.append(asyncio.create_task(runner.run()))
        return SessionHandles(
            respond=_make_respond(built.meeting_gate, worker),
            speak=_make_speak(worker),
            mute_toggle=_make_mute_toggle(built.mic_gate),
            clear=_make_clear(built.context, built.system_content),
        )

    app = ParrotboxApp(
        controls=controls,
        bus=bus,
        scenario_default=_scenario_from_argv(),
        reference_path=reference_path(),
        start_session=start_session,
        log_queue=log_queue,
        on_ready=on_ready,
    )
    try:
        await app.run_async()
    finally:
        # Best-effort graceful stop, then hard-exit. The TTS ThreadPoolExecutor
        # and sounddevice streams are non-daemon threads (a worker may be stuck
        # inside an uninterruptible torch generate()), so once the UI is gone
        # they'd keep the process alive forever. os._exit guarantees the app
        # actually quits when you press 'q'.
        for runner in runners:
            try:
                await asyncio.wait_for(runner.cancel(), timeout=3)
            except Exception:
                logger.exception("error cancelling runner on shutdown")
        for task in run_tasks:
            task.cancel()
        logger.complete()  # flush the file sink before we exit hard
        os._exit(0)


async def main():
    # The Textual TUI is the default; --console opts into the classic
    # prompts + log-stream flow. (--tui is still accepted as an explicit no-op.)
    console = "--console" in sys.argv
    _check_prerequisites(console=console)
    if console:
        await run_cli()
    else:
        await run_tui()


if __name__ == "__main__":
    asyncio.run(main())
