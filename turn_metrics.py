"""Per-turn latency instrumentation.

ARCHITECTURE.md asks for one stitched line per turn joining
`vad_stop → transcript → llm_first_token → tts_first_audio → playback`, plus
per-chunk RTF over session time to catch thermal throttling. This module
produces both, at INFO so they survive the default TUI log sink.

Two pieces, because the interesting numbers come from two places:

  * `TurnMetricsObserver` — a pipecat `BaseObserver`, so it watches every frame
    without inserting a processor into the pipeline. It stitches the turn
    timeline from the frames that mark each stage boundary.
  * `TTSChunkLog` — filled by ChatterboxTTSService, which is the only place
    that knows per-chunk synthesis cost. It also derives the metric that
    matters most for perceived responsiveness: **playback gaps**.

Why gaps: playback is continuous only while
`synth(chunk k) <= duration(chunk k-1)`. At the measured Chatterbox-Turbo MPS
RTF of ~0.9 a chunk costs roughly its own audio duration to make, so each chunk
can only be ~11% longer than its predecessor before the speaker runs dry
mid-sentence. A short first chunk followed by a full-sentence second chunk is
the classic offender. `gap_secs` measures that dead air directly instead of
inferring it from RTF.

Observer callbacks arrive on a proxy queue, so they are asynchronous relative to
the frame itself — always use `data.timestamp` (monotonic ns since pipeline
start), never a fresh clock read.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection

NS = 1_000_000_000


@dataclass
class ChunkStat:
    """One TTS synthesis chunk, with the dead air it caused (if any)."""

    index: int
    chars: int
    synth_secs: float
    audio_secs: float
    gap_secs: float  # silence before this chunk could start playing
    text: str

    @property
    def rtf(self) -> float:
        return self.synth_secs / max(self.audio_secs, 0.01)

    def as_dict(self) -> dict:
        return {
            "i": self.index,
            "chars": self.chars,
            "synth_ms": round(self.synth_secs * 1000),
            "audio_ms": round(self.audio_secs * 1000),
            "rtf": round(self.rtf, 2),
            "gap_ms": round(self.gap_secs * 1000),
            "text": self.text[:80],
        }


class TTSChunkLog:
    """Accumulates per-chunk synth stats for the current response.

    Shared by reference between ChatterboxTTSService (writer) and
    TurnMetricsObserver (drains it when the bot stops speaking). Both run on the
    pipeline event loop, so no locking is needed.
    """

    def __init__(self) -> None:
        self._chunks: list[ChunkStat] = []
        self._playback_end: float | None = None

    def reset(self) -> None:
        self._chunks.clear()
        self._playback_end = None

    def record(self, *, text: str, synth_secs: float, audio_secs: float, t_done: float) -> ChunkStat:
        """Log one finished synthesis. `t_done` is a monotonic timestamp.

        The gap model: chunk k cannot start playing until it exists (`t_done`),
        and will not start before its predecessor finishes playing. Any daylight
        between those two is silence the user hears.
        """
        if self._playback_end is None:
            gap = 0.0  # first chunk of a response — that latency is TTFA, not a gap
            playback_start = t_done
        else:
            gap = max(0.0, t_done - self._playback_end)
            playback_start = max(t_done, self._playback_end)
        self._playback_end = playback_start + audio_secs

        stat = ChunkStat(
            index=len(self._chunks),
            chars=len(text),
            synth_secs=synth_secs,
            audio_secs=audio_secs,
            gap_secs=gap,
            text=text,
        )
        self._chunks.append(stat)
        return stat

    def drain(self) -> list[ChunkStat]:
        chunks = list(self._chunks)
        self.reset()
        return chunks


@dataclass
class _Turn:
    """Timeline for one turn. All times are monotonic seconds since pipeline start."""

    t_user_stop: float | None = None
    t_transcript: float | None = None
    t_llm_start: float | None = None
    t_llm_first_token: float | None = None
    t_first_audio: float | None = None
    transcript: str = ""
    interrupted: bool = False
    chunks: list[ChunkStat] = field(default_factory=list)

    def _delta(self, a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return b - a

    def as_dict(self) -> dict:
        def ms(v: float | None) -> float | None:
            return None if v is None else round(v * 1000)

        gaps = [c.gap_secs for c in self.chunks]
        audio_total = sum(c.audio_secs for c in self.chunks)
        return {
            "ttfa_ms": ms(self._delta(self.t_user_stop, self.t_first_audio)),
            "stt_ms": ms(self._delta(self.t_user_stop, self.t_transcript)),
            "llm_ttft_ms": ms(self._delta(self.t_llm_start, self.t_llm_first_token)),
            "tts_first_ms": ms(self._delta(self.t_llm_first_token, self.t_first_audio)),
            "chunks": len(self.chunks),
            "audio_ms": round(audio_total * 1000),
            "gap_total_ms": round(sum(gaps) * 1000),
            "gap_max_ms": round(max(gaps) * 1000) if gaps else 0,
            "interrupted": self.interrupted,
            "transcript": self.transcript[:200],
            "chunk_detail": [c.as_dict() for c in self.chunks],
        }

    def summary(self) -> str:
        d = self.as_dict()

        def fmt(key: str) -> str:
            v = d[key]
            return "–" if v is None else f"{v / 1000:.2f}s"

        line = (
            f"TURN ttfa={fmt('ttfa_ms')} "
            f"(stt {fmt('stt_ms')} | llm {fmt('llm_ttft_ms')} | tts {fmt('tts_first_ms')}) "
            f"chunks={d['chunks']} audio={fmt('audio_ms')} "
            f"gaps={fmt('gap_total_ms')} max={fmt('gap_max_ms')}"
        )
        if d["interrupted"]:
            line += " [interrupted]"
        return line


class TurnMetricsObserver(BaseObserver):
    """Stitches one latency line per turn and writes it to the log (+ optional JSONL).

    Registered via `PipelineWorker(observers=[...])`. Only DOWNSTREAM frames are
    considered and every frame is deduped by id, because `on_push_frame` fires
    once per processor hop — the same TranscriptionFrame is seen several times
    on its way through the pipeline.
    """

    def __init__(self, chunk_log: TTSChunkLog, *, jsonl_path: Path | None = None) -> None:
        super().__init__()
        self._chunk_log = chunk_log
        self._jsonl_path = jsonl_path
        self._turn = _Turn()
        self._seen: deque[int] = deque(maxlen=2000)
        self._seen_set: set[int] = set()

    def _is_new(self, frame_id: int) -> bool:
        """Dedup across processor hops, with a bounded window."""
        if frame_id in self._seen_set:
            return False
        if len(self._seen) == self._seen.maxlen:
            self._seen_set.discard(self._seen[0])
        self._seen.append(frame_id)
        self._seen_set.add(frame_id)
        return True

    async def on_push_frame(self, data: FramePushed) -> None:
        if data.direction != FrameDirection.DOWNSTREAM:
            return
        frame = data.frame
        if not self._is_new(frame.id):
            return

        now = data.timestamp / NS
        turn = self._turn

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # New turn. stop_secs back-dates to when the user actually went
            # quiet, rather than when the VAD got around to saying so.
            self._turn = _Turn(t_user_stop=now - (frame.stop_secs or 0.0))
            self._chunk_log.reset()
        elif isinstance(frame, TranscriptionFrame):
            if turn.t_transcript is None:
                turn.t_transcript = now
                turn.transcript = (frame.text or "").strip()
        elif isinstance(frame, LLMFullResponseStartFrame):
            if turn.t_llm_start is None:
                turn.t_llm_start = now
        elif isinstance(frame, LLMTextFrame):
            if turn.t_llm_first_token is None:
                turn.t_llm_first_token = now
        elif isinstance(frame, BotStartedSpeakingFrame):
            if turn.t_first_audio is None:
                turn.t_first_audio = now
        elif isinstance(frame, InterruptionFrame):
            turn.interrupted = True
            self._emit()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._emit()

    def _emit(self) -> None:
        turn = self._turn
        turn.chunks = self._chunk_log.drain()
        # Nothing spoken and nothing heard — an empty barge-in or a stray
        # start/stop pair. Not worth a line.
        if turn.t_first_audio is None and not turn.chunks:
            self._turn = _Turn()
            return

        logger.info(turn.summary())
        if self._jsonl_path is not None:
            try:
                with self._jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(turn.as_dict()) + "\n")
            except OSError:
                logger.exception("could not write turn metrics JSONL")
        self._turn = _Turn()
