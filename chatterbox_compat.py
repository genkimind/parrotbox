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
