"""Chatterbox-Turbo text-to-speech for Pipecat, cloned from a reference recording.

Chatterbox has no streaming synthesis API — each chunk is a blocking generate()
call on MPS, run in a dedicated single worker thread (serializing synth jobs).
Latency is managed upstream by ClauseEagerAggregator chunking, not here.

Interruption model: the base TTSService cancels the task consuming run_tts()
when an InterruptionFrame arrives. An MPS kernel in flight cannot be cancelled —
the orphaned generate() finishes in its worker thread and the result is
discarded because nothing consumes the generator anymore. Never wait on the
worker from the event loop; abandoning the job is the design.
"""

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, StartFrame
from pipecat.services.tts_service import TTSService

from clause_aggregator import ClauseEagerAggregator
from runtime_controls import RuntimeControls

SLICE_SECS = 0.04  # small frames so the output transport can flush mid-chunk on barge-in


class ChatterboxTTSService(TTSService):
    def __init__(
        self,
        *,
        reference_wav: str | Path,
        device: str = "mps",
        controls: RuntimeControls | None = None,
        **kwargs,
    ):
        from pipecat.services.settings import TTSSettings

        super().__init__(
            settings=TTSSettings(model="chatterbox-turbo", voice=str(reference_wav), language=None),
            **kwargs,
        )
        self._reference_wav = Path(reference_wav)
        self._device = device
        # Live synthesis knobs (exaggeration/temperature/cfg_weight, and the
        # post-synth pitch-preserving speed stretch). Read fresh on every
        # _synth() call so the TUI's sliders take effect within a sentence.
        self._controls = controls or RuntimeControls()
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._text_aggregator = ClauseEagerAggregator()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._model is None:
            if not self._reference_wav.exists():
                raise FileNotFoundError(
                    f"Voice reference not found: {self._reference_wav} — "
                    "record one (see ARCHITECTURE.md, 'Voice clone reference capture')"
                )
            await asyncio.get_running_loop().run_in_executor(self._executor, self._load)

    def _load(self):
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        from chatterbox_compat import apply_mps_float64_patch, neutralize_tqdm_mp_lock

        apply_mps_float64_patch()
        neutralize_tqdm_mp_lock()  # else tqdm's mp-lock spawn crashes under the TUI

        t0 = time.monotonic()
        try:
            self._model = ChatterboxTurboTTS.from_pretrained(device=self._device)
            # Cache voice conditioning once; re-encoding the reference per
            # generate() call would add hundreds of ms to every chunk.
            self._model.prepare_conditionals(str(self._reference_wav))
            self._synth("Warm up.")  # pays MPS shader compile up front
        except Exception:
            if self._device == "mps":
                logger.exception(
                    "Chatterbox-Turbo failed on MPS (known early-version float64 "
                    "dtype issue) — retrying on CPU. Expect higher TTS latency."
                )
                self._device = "cpu"
                self._model = ChatterboxTurboTTS.from_pretrained(device="cpu")
                self._model.prepare_conditionals(str(self._reference_wav))
                self._synth("Warm up.")
            else:
                raise
        logger.info(
            f"Chatterbox-Turbo loaded+warm on {self._device} in {time.monotonic() - t0:.1f}s"
        )

    def _synth(self, text: str) -> bytes:
        c = self._controls.snapshot()  # one consistent read for this chunk
        wav = self._model.generate(
            text,
            exaggeration=c.exaggeration,
            temperature=c.temperature,
            cfg_weight=c.cfg_weight,
        )
        wav = wav.squeeze().cpu().numpy()
        if c.speed != 1.0:
            import pyrubberband

            # Rubber Band is a time-domain stretcher, so it doesn't smear phase
            # the way librosa's phase vocoder did — no reverby/echoey artifact on
            # voiced speech. rate > 1.0 speeds up (same sense as before). Shells
            # out to the `rubberband` binary per chunk, so it only runs when the
            # speed slider is off 1.0.
            wav = pyrubberband.time_stretch(wav, self._model.sr, c.speed)
        return (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        if self._model is None:
            yield ErrorFrame("Chatterbox model not loaded")
            return
        try:
            await self.start_tts_usage_metrics(text)
            t0 = time.monotonic()
            pcm = await asyncio.get_running_loop().run_in_executor(
                self._executor, self._synth, text
            )
            synth_secs = time.monotonic() - t0
            audio_secs = len(pcm) / 2 / self._model.sr
            logger.debug(
                f"TTS chunk {synth_secs * 1000:.0f}ms for {audio_secs:.1f}s audio "
                f"(RTF {synth_secs / max(audio_secs, 0.01):.2f}): {text[:60]!r}"
            )

            async def slices() -> AsyncIterator[bytes]:
                step = int(self._model.sr * SLICE_SECS) * 2
                for i in range(0, len(pcm), step):
                    yield pcm[i : i + step]

            async for tts_frame in self._stream_audio_frames_from_iterator(
                slices(), in_sample_rate=self._model.sr, context_id=context_id
            ):
                await self.stop_ttfb_metrics()
                yield tts_frame
        except Exception as e:
            logger.exception("Chatterbox synthesis failed")
            yield ErrorFrame(f"Chatterbox error: {e}")
        finally:
            await self.stop_ttfb_metrics()
