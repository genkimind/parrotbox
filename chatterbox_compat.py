"""MPS compatibility patch for chatterbox-tts 0.1.7.

`ChatterboxTurboTTS.prepare_conditionals` crashes on MPS with "Cannot convert a
MPS Tensor to float64": pyloudnorm's integrated_loudness returns a numpy
float64 scalar, and `wav * gain` inside norm_loudness promotes the float32
waveform to float64, which then rides through resample into the s3tokenizer
and voice encoder. MPS has no float64. Casting norm_loudness output back to
float32 fixes every downstream path at the single point of origin.

Remove once upstream fixes it (check on chatterbox-tts version bumps).
"""

import functools

import numpy as np


def apply_mps_float64_patch():
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    if getattr(ChatterboxTurboTTS.norm_loudness, "_f32_patched", False):
        return

    original = ChatterboxTurboTTS.norm_loudness

    @functools.wraps(original)
    def norm_loudness_f32(self, wav, sr, target_lufs=-27):
        out = original(self, wav, sr, target_lufs)
        return np.asarray(out, dtype=np.float32)

    norm_loudness_f32._f32_patched = True
    ChatterboxTurboTTS.norm_loudness = norm_loudness_f32


def neutralize_tqdm_mp_lock():
    """Stop tqdm from creating a multiprocessing lock.

    Chatterbox drives generation with `tqdm(range(...))`, and tqdm.__new__
    acquires a shared lock — building a multiprocessing RLock the first time,
    *even when the bar is disabled* (we set TQDM_DISABLE=1). Creating that lock
    spawns a resource_tracker subprocess. Under the Textual TUI, stdout's file
    descriptor has been replaced, so the spawn inherits an invalid fd (-1) and
    dies with "bad value(s) in fds_to_keep", taking TTS down with it.

    Forcing tqdm to use a plain threading lock avoids the subprocess entirely.
    Harmless outside the TUI (the bar is disabled either way).
    """
    import threading

    from tqdm.std import tqdm as std_tqdm

    std_tqdm.set_lock(threading.RLock())
