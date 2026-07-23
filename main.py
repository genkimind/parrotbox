"""Local voice agent: mic → Parakeet STT → DeepSeek → Chatterbox clone → speaker.

See ARCHITECTURE.md for the full design. Run with:

    DEEPSEEK_API_KEY=... uv run python main.py

Add --tui for the Textual interface: on-screen setup, a live conversation
pane, and sliders for exaggeration/temperature/cfg_weight/speed that take
effect mid-session. Logs go to logs/ instead of the console in that mode.

    DEEPSEEK_API_KEY=... uv run python main.py --tui
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Chatterbox runs tqdm progress bars inside every generate() call; in a
# long-running agent that's pure log spam. Must be set before chatterbox loads.
os.environ.setdefault("TQDM_DISABLE", "1")

from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.frames.frames import LLMRunFrame
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
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from meeting_mode import MeetingGate, install_key_listener
from persona import build_system_prompt
from runtime_controls import RuntimeControls
from stt_parakeet import ParakeetMLXSTTService
from transcript_tap import TranscriptTap
from tts_chatterbox import ChatterboxTTSService
from tts_normalize import TTSNormalizeFilter

REFERENCE_WAV = Path(os.environ.get("VOICE_REFERENCE", "voice_ref/reference.wav"))

# 1.0 is natural rate. Speeds above 1.0 play faster but can sound echoey.
DEFAULT_SPEED = 1.0


@dataclass
class SessionConfig:
    scenario: str | None = None
    speed: float = DEFAULT_SPEED
    half_duplex: bool = False
    meeting: bool = False


def _device_index(env_name: str) -> int | None:
    value = os.environ.get(env_name)
    return int(value) if value else None


def _scenario_from_argv() -> str | None:
    """Command-line scenario words, ignoring flags like --tui."""
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
) -> tuple[PipelineWorker, MeetingGate | None]:
    config = config or SessionConfig()
    # Seed the live controls from the session's starting speed; the TUI mutates
    # this same object in place while the pipeline runs.
    controls = controls or RuntimeControls(speed=config.speed)
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

    stt = ParakeetMLXSTTService()

    llm = OpenAILLMService(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        settings=OpenAILLMService.Settings(
            model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
            # Flash is a reasoning model; thinking tokens add ~350ms before the
            # first content token (measured), worth nothing in a voice turn.
            extra={"extra_body": {"thinking": {"type": "disabled"}}},
        ),
    )

    tts = ChatterboxTTSService(
        reference_wav=REFERENCE_WAV,
        controls=controls,
        # Rewrites LLM output into speakable form ("..." would be spoken as
        # "dot dot dot") — filtered text is also what history records.
        text_filters=[TTSNormalizeFilter()],
    )

    # Half-duplex for open speakers: LocalAudioTransport has no echo
    # cancellation, so without headphones the mic hears the speakers and the
    # agent barge-ins on itself, then converses with its own echo. Half duplex
    # mutes the mic while the bot talks — trade: no barge-in.
    mute_strategies = [AlwaysUserMuteStrategy()] if config.half_duplex else []

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": build_system_prompt(config.scenario, meeting=config.meeting),
            }
        ]
    )
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

    stages = [
        transport.input(),
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
        idle_timeout_secs=None,  # personal always-on session; never self-cancel
    )
    return worker, gate


def _check_prerequisites():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        logger.error("DEEPSEEK_API_KEY is not set")
        sys.exit(1)
    if not REFERENCE_WAV.exists():
        logger.error(
            f"Voice reference missing: {REFERENCE_WAV} — "
            "run scripts/capture_reference.py first (see README.md)"
        )
        sys.exit(1)


def _make_respond(gate: MeetingGate | None, worker: PipelineWorker):
    """A press-'r' handler for meeting mode: release one held reply."""

    def on_r():
        if gate is None:
            return
        if gate.held == 0:
            logger.info("meeting mode: nothing new since last reply")
            return
        gate.arm()
        asyncio.ensure_future(worker.queue_frame(LLMRunFrame()))

    return on_r


async def run_cli():
    config = get_session_config()
    logger.info(
        f"Session: scenario={config.scenario or '(none)'} speed={config.speed} "
        f"mode={'half' if config.half_duplex else 'full'}-duplex"
    )

    worker, gate = build_pipeline(config)
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


async def run_tui():
    import queue
    from datetime import datetime

    from tui_app import ParrotboxApp, SessionSetup, TranscriptBus

    # The TUI owns the screen, so logs never go to stderr. Two sinks: a
    # persistent file, and a DEBUG queue the boot screen tails during startup
    # (so the loading logs land in the conversation box) until the pipeline is
    # ready, at which point on_ready() detaches it.
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    logfile = logs_dir / f"parrotbox-{datetime.now():%Y%m%d-%H%M%S}.log"
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
    runners: list[WorkerRunner] = []
    run_tasks: list[asyncio.Task] = []

    async def start_session(setup: SessionSetup):
        config = SessionConfig(
            scenario=setup.scenario,
            half_duplex=setup.half_duplex,
            meeting=setup.meeting,
        )
        logger.info(f"Session: {config}")
        worker, gate = build_pipeline(config, controls=controls, bus=bus)
        runner = WorkerRunner(handle_sigint=False)  # Textual owns the keyboard
        await runner.add_workers(worker)
        runners.append(runner)
        run_tasks.append(asyncio.create_task(runner.run()))
        return _make_respond(gate, worker)

    app = ParrotboxApp(
        controls=controls,
        bus=bus,
        scenario_default=_scenario_from_argv(),
        start_session=start_session,
        log_queue=log_queue,
        on_ready=on_ready,
    )
    try:
        await app.run_async()
    finally:
        # Stop the pipeline so its audio threads don't keep the process alive.
        for runner in runners:
            try:
                await runner.cancel()
            except Exception:
                logger.exception("error cancelling runner on shutdown")
        for task in run_tasks:
            task.cancel()


async def main():
    _check_prerequisites()
    if "--tui" in sys.argv:
        await run_tui()
    else:
        await run_cli()


if __name__ == "__main__":
    asyncio.run(main())
