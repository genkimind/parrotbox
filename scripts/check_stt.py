"""Build-order step 2: validate Parakeet STT in isolation.

Synthesizes a known sentence with macOS `say`, converts to 16 kHz mono WAV,
transcribes it with parakeet-mlx, and reports accuracy + warm latency.

    uv run python scripts/check_stt.py                    # production gate (batch)
    uv run python scripts/check_stt.py --streaming        # + streaming re-eval bench
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


def bench_streaming(model, audio: np.ndarray) -> None:
    """Re-evaluation harness for streaming STT — NOT the production path.

    ARCHITECTURE.md "Streaming STT — evaluated, not shipped" records why
    (2026-08-21, parakeet-mlx 0.5.2): each add_audio re-decodes the draft
    region, so finalize ≈ one full-window pass and small attention contexts
    that shrink it cost visible WER on real voice. Re-run this when
    parakeet-mlx ships cheaper incremental decoding: ship streaming only if
    last-chunk cost drops well under the batch number at batch-equal text.
    """
    import mlx.core as mx

    chunk = int(0.32 * 16000)
    for ctx in ((256, 256), (64, 64)):
        times = []
        with model.transcribe_stream(context_size=ctx) as stream:
            for i in range(0, len(audio), chunk):
                c = audio[i:i + chunk]
                if len(c) < chunk:
                    c = np.pad(c, (0, chunk - len(c)))
                t0 = time.monotonic()
                stream.add_audio(mx.array(c))
                times.append(time.monotonic() - t0)
            text = stream.result.text
        print(f"stream ctx={ctx}: last-chunk {times[-1] * 1000:.0f}ms (finalize "
              f"proxy), RTF {sum(times) / (len(audio) / 16000):.2f} -> {text!r}")


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
    import wave as wavemod

    import mlx.core as mx

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

    if "--streaming" in sys.argv:
        bench_streaming(model, audio)
        print()
    ok = "language model" in text.lower() and "macbook" in text.lower()
    print("PASS" if ok else "FAIL — transcript doesn't match expected content")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
