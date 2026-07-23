"""Build-order step 2: validate Parakeet STT in isolation.

Synthesizes a known sentence with macOS `say`, converts to 16 kHz mono WAV,
transcribes it with parakeet-mlx, and reports accuracy + warm latency.

    uv run python scripts/check_stt.py
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

TEST_SENTENCE = (
    "I want to run the speech model locally on the MacBook and keep the "
    "language model on a hosted endpoint."
)


def main():
    tmp = Path(tempfile.mkdtemp())
    aiff = tmp / "test.aiff"
    wav = tmp / "test.wav"
    subprocess.run(["say", "-o", str(aiff), TEST_SENTENCE], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
        check=True,
    )

    from parakeet_mlx import from_pretrained

    t0 = time.monotonic()
    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    print(f"model load: {time.monotonic() - t0:.1f}s")

    # Cold + warm timings via the same raw-array path the Pipecat service uses.
    import mlx.core as mx
    import wave as wavemod

    with wavemod.open(str(wav)) as f:
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0

    for label in ("cold", "warm", "warm"):
        t0 = time.monotonic()
        with model.transcribe_stream() as stream:
            stream.add_audio(mx.array(audio))
            text = stream.result.text
        print(f"{label}: {(time.monotonic() - t0) * 1000:.0f}ms for "
              f"{len(audio) / 16000:.1f}s audio -> {text!r}")

    ok = "language model" in text.lower() and "macbook" in text.lower()
    print("PASS" if ok else "FAIL — transcript doesn't match expected content")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
