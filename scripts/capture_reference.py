"""Build-order step 4 (part 2): record voice-clone reference audio.

Records 90 seconds from the default mic at 48 kHz mono WAV, unprocessed.
Before recording: quiet damped room, disable macOS Voice Isolation (Control
Center > Mic Mode > Standard), talk naturally TO someone — include a question
and a plain statement, normal energy, no reading voice.

Then slice candidate 12s windows and pick with scripts/eval_clone.py.

    uv run python scripts/capture_reference.py [seconds]
"""

import sys
import wave
from pathlib import Path

import pyaudio

OUT = Path("voice_ref/capture_full.wav")
RATE = 48000


def main():
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    OUT.parent.mkdir(exist_ok=True)
    pa = pyaudio.PyAudio()
    info = pa.get_default_input_device_info()
    print(f"recording {seconds}s from: {info['name']}")
    print("talk naturally — question, statement, normal energy. Ctrl-C to stop early.")
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                     frames_per_buffer=4800)
    frames = []
    try:
        for i in range(seconds * 10):
            frames.append(stream.read(4800, exception_on_overflow=False))
            if i % 10 == 0:
                print(f"\r{i // 10}s", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    with wave.open(str(OUT), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(b"".join(frames))
    print(f"\nwrote {OUT} ({len(frames) * 0.1:.0f}s)")
    print("next: cut SEVERAL 12s candidate windows from different parts (vary -ss):")
    print("  ffmpeg -i voice_ref/capture_full.wav -ss 5  -t 12 voice_ref/cand_a.wav")
    print("  ffmpeg -i voice_ref/capture_full.wav -ss 30 -t 12 voice_ref/cand_b.wav")
    print("  ffmpeg -i voice_ref/capture_full.wav -ss 60 -t 12 voice_ref/cand_c.wav")
    print("then compare candidates (each gets 10 test sentences in scratch_clone_eval/):")
    print("  uv run --with resemblyzer python scripts/eval_clone.py "
          "--real voice_ref/capture_full.wav voice_ref/cand_*.wav")
    print("promote the winning CANDIDATE (not a test sentence):")
    print("  cp voice_ref/cand_<winner>.wav voice_ref/reference.wav")


if __name__ == "__main__":
    main()
