# clone2 — local voice agent in a cloned voice

A single-user voice agent that talks *as* a chosen persona, in a cloned voice,
running locally on a Mac. Mic → Parakeet STT (MLX) → DeepSeek → Chatterbox-Turbo
voice clone (MPS) → speaker, orchestrated by Pipecat. It listens on your mic and
answers out loud in a voice cloned from a short reference recording, staying in a
persona you define in `persona.md`.

Design and rationale: [ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

- **macOS on Apple Silicon** (M-series). STT (parakeet-mlx) and TTS
  (Chatterbox-Turbo on MPS) both use Apple's Metal stack; this will not run as-is
  on Intel Macs, Windows, or Linux.
- ~6 GB free memory while running; ~16 GB machine or larger recommended.
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- An OpenAI-compatible LLM endpoint. The default is DeepSeek — you need a
  `DEEPSEEK_API_KEY`. Any OpenAI-compatible provider works via `LLM_BASE_URL` /
  `LLM_MODEL` (see the bottom of this file).

## Setup

```bash
brew install portaudio ffmpeg
uv sync
```

## Two things you must provide

This repo ships with an **example persona** and **no voice reference** — the two
things that make the agent sound like a specific person are intentionally left
for you to supply:

1. A **persona** — who the agent is and how it talks (`persona.md`).
2. A **voice reference** — a short recording of the voice to clone
   (`voice_ref/reference.wav`).

### 1. Write the persona

The system prompt is loaded from a persona markdown file at startup and never
touched by code. The file that ships here, [`persona.md`](persona.md), is a
clearly-marked **example** (a fictional woodworker named Robin Vale) that
demonstrates the format.

**To use your own persona without committing it, put it in `persona.local.md`.**
That filename is gitignored, so your real persona stays on your machine and is
never pushed; the code prefers it over the example when present. This lets you
track this repo directly (pull code updates, push fixes) while keeping your
identity — and your `voice_ref/` recording, also gitignored — entirely local. No
second copy to maintain.

```bash
cp persona.md persona.local.md   # then edit persona.local.md with your own persona
```

A persona has two parts that matter most:

- **Who they are and how they talk** — background, register, verbal tics. Keep it
  concrete and behavioral ("get to the point, no pleasantries") rather than
  abstract.
- **Style samples** — six to ten short snippets of how the person actually
  speaks. This is the single biggest lever on how the agent sounds. Real
  transcript fragments work far better than invented ones. Six to ten short,
  varied samples beat twenty long ones.

  **Tip — capturing samples:** the easiest way to get natural samples is to just
  talk and let something transcribe you. [Ghost Pepper](https://github.com/matthartman/ghost-pepper)
  is a great on-device, local speech-to-text transcriber — highly recommended.
  Just talk and it transcribes your words to text as you speak; paste those
  transcripts straight into your Style samples.

Keep the structural sections of the example file as-is (Interruptions, Rhythm and
length, What not to do, The test) — those are what keep output speakable through
text-to-speech, and are not persona-specific.

The scenario you enter at startup (see Run) is appended to the end of this prompt
at runtime; the persona text itself stays byte-stable so the LLM's prefix cache
stays warm.

### 2. Record the voice reference

Chatterbox clones a voice from a single ~12-second reference clip. Capture a
longer sample, then cut and compare several candidate windows:

```bash
uv run python scripts/capture_reference.py         # 90s raw capture → voice_ref/capture_full.wav

# Cut SEVERAL 12s candidate windows from different parts of the capture (vary -ss).
# Chatterbox conditions the whole clone on whichever ~12s you give it, so different
# windows produce noticeably different clones — the point of this step is comparing them:
ffmpeg -i voice_ref/capture_full.wav -ss 5  -t 12 voice_ref/cand_a.wav
ffmpeg -i voice_ref/capture_full.wav -ss 30 -t 12 voice_ref/cand_b.wav
ffmpeg -i voice_ref/capture_full.wav -ss 60 -t 12 voice_ref/cand_c.wav

uv run --with resemblyzer python scripts/eval_clone.py --real voice_ref/capture_full.wav voice_ref/cand_*.wav

# eval output: scratch_clone_eval/<candidate>/0..9.wav are the ten fixed TEST
# SENTENCES synthesized from that candidate — you never pick one of those.
# Judge each CANDIDATE by its WER/similarity line + listening across its ten
# outputs, then promote the winning candidate:
cp voice_ref/cand_<winner>.wav voice_ref/reference.wav
```

Capture rules: quiet damped room, macOS mic mode set to Standard (no Voice
Isolation), talk naturally, include a question and a statement. Reverb in the
reference becomes permanent reverb in the clone.

> Only clone a voice you have the right to use — your own, or one you have clear
> permission to reproduce. `voice_ref/` is gitignored and never distributed.

## Validate the stages (optional, recommended build order)

```bash
uv run python scripts/check_stt.py                 # STT accuracy + latency (gate: <300ms warm)
DEEPSEEK_API_KEY=... uv run python scripts/measure_ttft.py   # LLM TTFT p50/p95 (gate: p95 <1s)
uv run python scripts/check_tts.py                 # TTS on MPS + RTF (gate: warm RTF <1.0)
```

## Run

```bash
DEEPSEEK_API_KEY=... uv run python main.py                        # prompts for a scenario
DEEPSEEK_API_KEY=... uv run python main.py daily scrum standup    # scenario from argv
```

Startup asks four questions (enter accepts the default on each):

1. **Scenario** — short context that shapes register and tone for the session
   ("scrum meeting", "chatting with a friend"); skipped if given on the command line
2. **Voice speed** — pitch-preserving playback rate (default 1.0 = natural).
   Values above 1.0 play faster but can sound echoey — nudge up carefully (1.05–1.2)
3. **Mic mode** — full duplex (barge-in, needs headphones) or half duplex
   (open speakers: mic muted while the agent talks, no barge-in — without this,
   the mic hears the speakers and the agent converses with its own echo)
4. **Response mode** — live (replies after every turn, the default) or meeting
   (transcribes silently; press `r` to get one reply over everything heard so far)

First start takes ~30s (model loads + MPS shader warmup).

### Meeting mode

In meeting mode the agent listens and accumulates everything into context but
says nothing until you press `r` (no Enter needed). It then responds once to
what's directly relevant in the backlog and goes back to holding. Pressing `r`
with nothing new logs "nothing new" and stays silent. Timing details: an
utterance still in progress when you press `r` lands in the *next* reply, and
pressing `r` while the agent is talking queues the reply behind the current
one. Half duplex + meeting works but mutes the mic while the agent talks, so
meeting speech during a reply is lost — full duplex with headphones is the
better meeting setup.

### Echo / headphones

There is no echo cancellation. **Headphones are required for barge-in** (full
duplex). On open speakers, pick half duplex at the mic-mode question — otherwise
the mic hears the agent, barge-in triggers on its own voice, and it ends up
conversing with its own echo (cut-off babble every few seconds).

## Configuration (environment variables)

- `DEEPSEEK_API_KEY` — required for the default DeepSeek endpoint.
- `LLM_MODEL`, `LLM_BASE_URL` — point at any OpenAI-compatible provider/model.
  Defaults target DeepSeek.
- `VOICE_REFERENCE` — path to the reference WAV (default `voice_ref/reference.wav`).
- `AUDIO_IN_DEVICE` / `AUDIO_OUT_DEVICE` — device indices for routing to a virtual
  device (e.g. BlackHole, for feeding other apps like Zoom); enumerate with
  `uv run python scripts/list_audio_devices.py`.

## Notes

This is a personal, single-user, hobby-scale project, not a hardened product.
It targets Apple Silicon specifically. The bundled `persona.md` and the eval test
sentences are illustrative examples — replace them with your own. You are
responsible for the licenses of the models and libraries it pulls in, and for
having the right to clone whatever voice you use.
