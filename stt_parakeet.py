"""Parakeet TDT speech-to-text for Pipecat via parakeet-mlx (Apple Silicon / Metal).

Segmented mode: the base class buffers mic audio between VAD start/stop and hands
the completed utterance to run_stt(). Parakeet transcribes ~5s of audio in
100-300ms warm on M-series, so whole-segment transcription keeps this simple.
If finalize latency shows up in the per-turn metrics, the upgrade path is a
persistent model.transcribe_stream() context fed during speech.
"""

import asyncio
import time
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, StartFrame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


class ParakeetMLXSTTService(SegmentedSTTService):
    def __init__(self, *, model_name: str = DEFAULT_MODEL, **kwargs):
        from pipecat.services.settings import STTSettings

        # language=None: Parakeet v3 auto-detects; not runtime-switchable here.
        super().__init__(settings=STTSettings(model=model_name, language=None), **kwargs)
        self._model_name = model_name
        self._model = None

    def can_generate_metrics(self) -> bool:
        # Without this the start/stop_processing_metrics() calls in run_stt are
        # no-ops (FrameProcessor defaults to False), and pipecat never reports
        # this service's TTFB.
        return True

    @property
    def wants_wav_segments(self) -> bool:
        # Raw 16-bit PCM, not a WAV container.
        return False

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._model is None:
            await asyncio.to_thread(self._load)

    def _load(self):
        from parakeet_mlx import from_pretrained

        t0 = time.monotonic()
        self._model = from_pretrained(self._model_name)
        # Warmup so the first real utterance doesn't pay lazy-compile cost.
        self._transcribe(np.zeros(16000, dtype=np.float32))
        logger.info(f"Parakeet {self._model_name} loaded+warm in {time.monotonic() - t0:.1f}s")

    def _transcribe(self, audio_f32: np.ndarray) -> str:
        import mlx.core as mx

        # A short-lived streaming context accepts raw arrays directly, unlike
        # transcribe() which wants a file path.
        with self._model.transcribe_stream() as stream:
            stream.add_audio(mx.array(audio_f32))
            return stream.result.text

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if self._model is None:
            yield ErrorFrame("Parakeet model not loaded")
            return
        try:
            await self.start_processing_metrics()
            t0 = time.monotonic()
            audio_f32 = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            text = await asyncio.to_thread(self._transcribe, audio_f32)
            elapsed_ms = (time.monotonic() - t0) * 1000
            await self.stop_processing_metrics()
            logger.debug(
                f"STT finalize {elapsed_ms:.0f}ms for {len(audio_f32) / 16000:.1f}s audio: {text!r}"
            )
            text = text.strip()
            if text:
                yield TranscriptionFrame(text, self._user_id, time_now_iso8601())
        except Exception as e:
            logger.exception("Parakeet transcription failed")
            yield ErrorFrame(f"Parakeet error: {e}")
