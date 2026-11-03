"""Voice-clone reference workflow: record, slice candidates, preview, promote.

This is the logic behind the TUI's Clone screen (tui_app.CloneScreen) — kept
UI-free so it is testable headless and reusable from scripts. It replaces the
old manual flow (capture_reference.py + ffmpeg window cuts + eval_clone.py +
cp) with: record once, auto-slice spread candidate windows, synthesize one
fixed preview sentence per candidate with a single Chatterbox load, listen,
promote the winner to reference.wav.

The objective path (Parakeet WER + resemblyzer similarity over a 10-sentence
set) is still available offline via scripts/eval_clone.py for fine-tuning a
final pick — see README.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import wave
from pathlib import Path

import numpy as np
from loguru import logger

RATE = 48000  # capture rate: unprocessed mono WAV, matching scripts/capture_reference.py
WINDOW_SECS = 12  # Chatterbox conditions the clone on ~this much audio
MIN_CAPTURE_SECS = 15  # shortest capture that still yields one full window
N_CANDIDATES = 4  # windows spread across the capture; different windows clone differently
EDGE_SECS = 2.0  # skip the very start/end (mic handling noise, trailing silence)

# One fixed sentence for every candidate: a question and a statement, like the
# capture advice, so A/B listening compares delivery not content.
PREVIEW_TEXT = (
    "Hey, it's me — does this sound about right? "
    "I want the pauses and the line endings to land naturally."
)


def reference_path() -> Path:
    """The live clone reference. Read at call time so tests and the TUI can
    relocate it via VOICE_REFERENCE."""
    return Path(os.environ.get("VOICE_REFERENCE", "voice_ref/reference.wav"))


def capture_path() -> Path:
    return reference_path().parent / "capture_full.wav"


def candidates_dir() -> Path:
    return reference_path().parent / "candidates"


def write_wav(path: Path, pcm: bytes, rate: int = RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm)


def write_wav_f32(path: Path, audio: np.ndarray, rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    write_wav(path, pcm, rate)


class Recorder:
    """Default-mic capture to 48 kHz mono WAV, driven from a reader thread.

    pyaudio's read() blocks, so it runs on its own thread; start()/stop() are
    called from the TUI event loop and never block on audio.
    """

    def __init__(self, rate: int = RATE) -> None:
        self._rate = rate
        self._frames: list[bytes] = []
        self._recording = False
        self._started_at = 0.0
        self._thread: threading.Thread | None = None
        self._stream = None
        self._pa = None

    @property
    def recording(self) -> bool:
        return self._recording

    def elapsed(self) -> float:
        return time.monotonic() - self._started_at if self._recording else 0.0

    def start(self) -> None:
        import pyaudio

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=4800,
        )
        self._frames = []
        self._recording = True
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while self._recording:
            try:
                self._frames.append(self._stream.read(4800, exception_on_overflow=False))
            except Exception:
                logger.exception("recorder read failed")
                break

    def stop(self) -> Path:
        """Stop capturing and write the WAV; returns its path."""
        self._recording = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa is not None:
            self._pa.terminate()
        out = capture_path()
        write_wav(out, b"".join(self._frames), self._rate)
        return out


def slice_candidates(
    capture: Path,
    out_dir: Path,
    *,
    n: int = N_CANDIDATES,
    window_secs: float = WINDOW_SECS,
) -> list[tuple[float, Path]]:
    """Cut `n` evenly-spread windows out of a capture, no ffmpeg involved.

    Returns (start_secs, path) per candidate. Windows are spread edge-to-edge
    so they sample different energy/register of the recording — Chatterbox
    conditions the whole clone on whichever window wins, so the spread is what
    makes the comparison meaningful.
    """
    with wave.open(str(capture), "rb") as f:
        rate = f.getframerate()
        total_frames = f.getnframes()
        pcm = f.readframes(total_frames)
    total_secs = total_frames / rate
    if total_secs < MIN_CAPTURE_SECS:
        raise ValueError(
            f"capture is {total_secs:.0f}s — need at least {MIN_CAPTURE_SECS}s of speech"
        )

    window_frames = int(window_secs * rate)
    first = min(EDGE_SECS, max(0.0, (total_secs - window_secs) / 2))
    last = max(first, total_secs - EDGE_SECS - window_secs)
    if n < 2 or last == first:
        starts = [first]
    else:
        starts = [first + (last - first) * i / (n - 1) for i in range(n)]

    out = []
    for i, start in enumerate(starts):
        frame0 = int(start * rate)
        chunk = pcm[frame0 * 2 : (frame0 + window_frames) * 2]
        path = out_dir / f"cand_{chr(ord('a') + i)}.wav"
        write_wav(path, chunk, rate)
        out.append((start, path))
    return out


class ClonePreviewer:
    """One Chatterbox load, reused across all candidate previews.

    Loading the model costs tens of seconds; preparing conditionals per
    candidate is cheap by comparison, so the whole preview batch shares a
    single instance. Call unload() when done — the session pipeline loads its
    own copy, and holding both doubles the memory footprint.
    """

    def __init__(self, device: str = "mps") -> None:
        self._device = device
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        from .chatterbox_compat import apply_mps_float64_patch, neutralize_tqdm_mp_lock

        apply_mps_float64_patch()
        neutralize_tqdm_mp_lock()  # else tqdm's mp-lock spawn crashes under the TUI

        t0 = time.monotonic()
        self._model = ChatterboxTurboTTS.from_pretrained(device=self._device)
        logger.info(f"Clone previewer loaded on {self._device} in {time.monotonic() - t0:.1f}s")

    def synthesize(self, candidate: Path, text: str = PREVIEW_TEXT) -> tuple[np.ndarray, int]:
        """Render the fixed preview sentence conditioned on one candidate."""
        if self._model is None:
            raise RuntimeError("load() first")
        self._model.prepare_conditionals(str(candidate))
        wav = self._model.generate(text).squeeze().cpu().numpy()
        return wav, self._model.sr

    def unload(self) -> None:
        """Free the model so a session started right after cloning doesn't
        carry two copies in memory."""
        self._model = None
        try:
            import torch

            if self._device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass


def promote(candidate: Path) -> Path:
    """Make a candidate the live reference. Overwrites the previous one."""
    dest = reference_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, dest)
    logger.info(f"voice reference updated: {candidate} -> {dest}")
    return dest
