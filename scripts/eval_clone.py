"""Build-order step 4 (part 3): pick the best clone reference objectively.

For each candidate reference WAV: synthesize a fixed sentence set, then score
  1. intelligibility — Parakeet WER of the TTS output against the known text
     (target < 5%)
  2. speaker similarity — resemblyzer cosine between TTS output and real
     recordings (optional: `uv run --with resemblyzer ...`). Calibrate against
     your real-vs-real self-similarity; pick the candidate closest to it.

    uv run python scripts/eval_clone.py voice_ref/cand_a.wav voice_ref/cand_b.wav
    uv run --with resemblyzer python scripts/eval_clone.py --real voice_ref/capture_full.wav voice_ref/cand_*.wav
"""

import re
import sys
import wave
from pathlib import Path

import numpy as np

EVAL_SENTENCES = [
    "The deployment failed because the service mesh sidecar never came up.",
    "Yeah, that's good, but I don't want to run it in the cloud.",
    "Can you explain how the scheduler decides which node gets the pod?",
    "First thing is latency, then there's memory, and the one that matters is both.",
    "I spent years working with my hands before I ever ran a business.",
    "What's the actual failure mode when the cache goes cold?",
    "That press release is doing a lot of work to avoid saying what broke.",
    "Kind of like a build system, except nobody owns it.",
    "Hold on, let me think about whether that's even the right question.",
    "We shipped it Friday and it was on fire by Monday morning.",
]


def wer(ref: str, hyp: str) -> float:
    r = re.sub(r"[^a-z' ]", "", ref.lower()).split()
    h = re.sub(r"[^a-z' ]", "", hyp.lower()).split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / max(len(r), 1)


def write_wav(path: Path, audio: np.ndarray, sr: int):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def main():
    args = sys.argv[1:]
    real_wav = None
    if "--real" in args:
        i = args.index("--real")
        real_wav = args[i + 1]
        args = args[:i] + args[i + 2:]
    candidates = [Path(a) for a in args]
    if not candidates:
        print(__doc__)
        sys.exit(1)

    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from parakeet_mlx import from_pretrained

    from parrotbox.chatterbox_compat import apply_mps_float64_patch

    apply_mps_float64_patch()
    tts = ChatterboxTurboTTS.from_pretrained(device="mps")
    stt = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

    encoder = None
    if real_wav:
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            encoder = VoiceEncoder()
            real_embed = encoder.embed_utterance(preprocess_wav(real_wav))
        except ImportError:
            print("resemblyzer not installed — skipping similarity "
                  "(rerun with: uv run --with resemblyzer ...)")

    out_root = Path("scratch_clone_eval")
    for cand in candidates:
        tts.prepare_conditionals(str(cand))
        out_dir = out_root / cand.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        wers, sims = [], []
        for i, text in enumerate(EVAL_SENTENCES):
            audio = tts.generate(text).squeeze().cpu().numpy()
            path = out_dir / f"{i}.wav"
            write_wav(path, audio, tts.sr)
            hyp = stt.transcribe(str(path)).text
            wers.append(wer(text, hyp))
            if encoder is not None:
                from resemblyzer import preprocess_wav
                embed = encoder.embed_utterance(preprocess_wav(str(path)))
                sims.append(float(np.dot(real_embed, embed)))
        line = f"{cand.name}: WER {np.mean(wers) * 100:.1f}%"
        if sims:
            line += f", similarity {np.mean(sims):.3f} (calibrate vs real-vs-real)"
        print(line)
    print(f"\naudio in {out_root}/<candidate>/ — also A/B by ear before deciding")


if __name__ == "__main__":
    main()
