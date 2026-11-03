"""Build-order step 4 (part 1): validate Chatterbox-Turbo on MPS and measure RTF.

Loads the model, optionally clones from a reference WAV (falls back to the
built-in voice so this runs before any reference is recorded), synthesizes test
sentences, reports per-sentence RTF, and writes the audio for listening.

    uv run python scripts/check_tts.py [path/to/reference.wav]
"""

import sys
import time
import wave
from pathlib import Path

import numpy as np

SENTENCES = [
    "Yeah, that's good.",
    "First thing is the speech model, then there's the language model, and the "
    "one that actually matters is the voice.",
    "I don't want to host the language model, but the speech stages could be local.",
]

OUT_DIR = Path("scratch_tts_check")


def main():
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    from parrotbox.chatterbox_compat import apply_mps_float64_patch

    apply_mps_float64_patch()
    ref = sys.argv[1] if len(sys.argv) > 1 else None
    t0 = time.monotonic()
    model = ChatterboxTurboTTS.from_pretrained(device="mps")
    print(f"model load (mps): {time.monotonic() - t0:.1f}s")

    if ref:
        t0 = time.monotonic()
        model.prepare_conditionals(ref)
        print(f"prepare_conditionals: {time.monotonic() - t0:.1f}s")
    else:
        print("no reference wav given — using built-in voice")

    OUT_DIR.mkdir(exist_ok=True)
    for i, text in enumerate(SENTENCES):
        t0 = time.monotonic()
        wav_tensor = model.generate(text)
        synth = time.monotonic() - t0
        audio = wav_tensor.squeeze().cpu().numpy()
        secs = len(audio) / model.sr
        label = "cold" if i == 0 else "warm"
        print(f"[{label}] {synth * 1000:.0f}ms for {secs:.1f}s audio "
              f"(RTF {synth / secs:.2f}): {text[:50]!r}")
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        out = OUT_DIR / f"sentence_{i}.wav"
        with wave.open(str(out), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(model.sr)
            f.writeframes(pcm.tobytes())
    print(f"\naudio written to {OUT_DIR}/ — listen with: afplay {OUT_DIR}/sentence_1.wav")
    print("Gate: warm RTF < 1.0 required (chunk n+1 must synthesize faster than "
          "chunk n plays); RTF < 0.5 is comfortable.")


if __name__ == "__main__":
    main()
