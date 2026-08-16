# parrotbox — local voice agent in a cloned voice

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
brew install portaudio rubberband
uv sync
```

(`rubberband` is the time-stretch engine used by the voice **Speed** control; a
phase vocoder was replaced with it because it left voiced speech sounding
echoey. Speed changes shell out to the `rubberband` binary, so it must be on
your `PATH`.)

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

### 2. Clone the voice (in the TUI)

Chatterbox clones a voice from a single ~12-second reference clip. The TUI
walks you through it: run `uv run python main.py`, click **Clone a voice…** on
the setup screen, then:

1. **Record** — one sample, 30s or more. Quiet damped room, macOS mic mode on
   Standard (no Voice Isolation), talk naturally: a question, a statement,
   normal energy. Reverb in the recording becomes permanent reverb in the
   clone.
2. **Build previews** — the app slices several candidate 12s windows spread
   across the capture (no ffmpeg — Chatterbox conditions the whole clone on
   whichever window it gets, and different windows clone noticeably
   differently, so the spread is the comparison), loads Chatterbox once, and
   synthesizes one fixed preview sentence per candidate.
3. **Listen and pick** — play each candidate, hit **use this voice** on the
   winner. It becomes `voice_ref/reference.wav` and the session uses it
   immediately — no restart.

The setup screen's **Start session** stays disabled until a reference exists,
so a first run can't fail deep inside model loading.

> Only clone a voice you have the right to use — your own, or one you have clear
> permission to reproduce. `voice_ref/` is gitignored and never distributed.

## Validate the stages (optional, recommended build order)

```bash
uv run python scripts/check_stt.py                 # STT accuracy + latency (gate: <300ms warm)
DEEPSEEK_API_KEY=... uv run python scripts/measure_ttft.py   # LLM TTFT p50/p95 (gate: p95 <1s)
uv run python scripts/check_tts.py                 # TTS on MPS + RTF (gate: warm RTF <1.0)
uv run python scripts/check_chunking.py            # chunk policy: correctness + gap estimate
```

`check_chunking.py` needs no models and runs in a second — see
[Latency and chunk tuning](#latency-and-chunk-tuning).

## Run

The Textual interface (setup on screen, a clean conversation pane instead of log
spam, and live voice sliders) is the **default** — see [TUI](#tui) below:

```bash
DEEPSEEK_API_KEY=... uv run python main.py                       # Textual interface (default)
DEEPSEEK_API_KEY=... uv run python main.py daily standup         # scenario from argv, skips that prompt
```

Prefer a plain console? Add `--console` for the classic prompts + log-stream flow:

```bash
DEEPSEEK_API_KEY=... uv run python main.py --console                       # prompts for a scenario
DEEPSEEK_API_KEY=... uv run python main.py --console daily scrum standup   # scenario from argv
```

In the classic console flow, startup asks four questions (enter accepts the default on each):

1. **Scenario** — short context that shapes register and tone for the session
   ("scrum meeting", "chatting with a friend"); skipped if given on the command line
2. **Voice speed** — pitch-preserving playback rate (default 1.0 = natural).
   Values above 1.0 play faster but can sound echoey — nudge up carefully (1.05–1.2)
3. **Mic mode** — full duplex (barge-in, needs headphones) or half duplex
   (open speakers: mic muted while the agent talks, no barge-in — without this,
   the mic hears the speakers and the agent converses with its own echo)
4. **Response mode** — live (replies after every turn, the default) or meeting
   (transcribes silently; press `r` to get one reply over everything heard so far)

First start takes ~1 min (model loads + MPS shader warmup).

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

### TUI

The default interface is a [Textual](https://textual.textualize.io) app (pass
`--console` to get the plain prompts + log stream instead). It has three parts:

- **Setup screen** — the same four questions (scenario, mic mode, response
  mode), on screen. Voice speed moves to a live slider, so it isn't asked here.
  A **Clone a voice…** button runs the record → preview → pick workflow (see
  above) right from setup; **Start session** stays disabled until a voice
  reference exists.
- **Conversation pane** — only what was said and what the agent spoke; no logs.
  (Logs are redirected to `logs/parrotbox-<timestamp>.log`.)
- **Voice sliders** — Exaggeration, Temperature, CFG weight, Speed, and Volume,
  adjustable *while talking*. Because Chatterbox takes these per synthesis call,
  a change is audible within a sentence. The sliders run to deliberately wide
  ceilings (well past the usual "safe" range) so there's headroom to push a
  voice hard — the upper reaches get extreme and can distort, which is the point.
  Volume defaults to 1.0 (100%) and goes up to 1.5, a +50% overboost for when
  full macOS volume still isn't loud enough (it hard-limits, so it distorts hot).

Keys: `↑`/`↓` pick a slider, `←`/`→` nudge it by 0.05 (`Shift` for a 0.50 jump),
`t` types a line for the voice to speak, `m` mutes/unmutes the mic, `c` clears
the conversation (press twice to confirm), `r` responds in meeting mode, `q`
quits.

Press `t` to type a line and have the clone speak it verbatim (Enter sends, Esc
cancels) — handy for saying something exact without going through the model. The
typed line shows in the conversation pane and is added to the session's context,
so the agent knows it said it for the rest of the run. It isn't saved between
runs.

`m` toggles mic mute — while muted the agent hears nothing (the header shows
🔇 mic muted). `c` starts a fresh conversation: it clears the pane and resets the
model's context to just the system prompt, so nothing said earlier is
remembered. Because that's destructive, it needs two presses — the first arms
it (Esc or a 3-second pause cancels), the second confirms.

Exaggeration is the emotion dial and the main knob — it has the thickest bar and
is marked "adjust this first." Raise it for a more excited, animated delivery
(the default 0.5 is a touch livelier than neutral); past ~0.7 it speeds up and
gets theatrical, and the high end of the slider goes further still. Temperature
and CFG weight are the secondary trims (shown as thinner bars — same length, just
lighter): if it rushes at high exaggeration, raise CFG weight to slow the pacing
back down.

### Echo / headphones

There is no echo cancellation. **Headphones are required for barge-in** (full
duplex). On open speakers, pick half duplex at the mic-mode question — otherwise
the mic hears the agent, barge-in triggers on its own voice, and it ends up
conversing with its own echo (cut-off babble every few seconds).

## Latency and chunk tuning

Every turn writes one latency line to the log, plus a machine-readable record to
`logs/turns-<timestamp>.jsonl`:

```
TURN ttfa=2.35s (stt 0.25s | llm 1.04s | tts 1.05s) chunks=3 audio=6.00s gaps=1.60s max=1.60s
```

`ttfa` is what you feel as "how long until it starts talking", broken into its
three stages. **`gaps` is dead air *after* it started talking** — the agent
running out of synthesized audio mid-reply and stalling before the next chunk is
ready. Those are two different complaints with two different fixes, which is why
they're measured separately.

Gaps happen because Chatterbox has no streaming synthesis: audio is made one
chunk at a time at RTF ~0.9, so a chunk costs roughly its own duration to
produce. Playback only stays continuous while each chunk lasts long enough to
cover the *next* one being synthesized — about a 1.1× growth budget per chunk.
`ClauseEagerAggregator` therefore ramps chunk sizes, sizing each one against
what was actually just emitted rather than emitting whole sentences.

Tune it without editing code (defaults shown):

- `PARROTBOX_MIN_FIRST_CHARS` (24) — first chunk floor. Sets time-to-first-audio.
  Raising it makes the opening phrase longer but delays every first word;
  measurements say it buys very little, because the ramp is what prevents gaps.
- `PARROTBOX_MIN_CHUNK_CHARS` (24) — prosody floor for later chunks. Chatterbox
  degrades badly below ~5 words, so don't go lower.
- `PARROTBOX_CHUNK_RAMP` (1.25) — per-chunk growth. Lower is safer (more, smaller
  chunks); higher gives better prosody but risks gaps.
- `PARROTBOX_MAX_CHUNK_CHARS` (180) — stop growing past this.

`scripts/check_chunking.py` estimates first-audio and gaps for a policy without
loading any models, so you can compare settings in seconds. Treat it as a guide:
`gap_total_ms` in `logs/turns-*.jsonl` is the ground truth. If gaps are near
zero and it still feels slow, the problem is `ttfa`, not chunking — check which
of the three stages dominates before changing anything here.

One case the ramp cannot fix is a long sentence with no commas: there is nothing
to split on, so it synthesizes as one long chunk. That needs streaming synthesis
(see ARCHITECTURE.md).

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
sentences are illustrative examples — replace them with your own.

## ⚠️ Voice cloning — your responsibility

This software clones a person's voice from a short reference recording. **Only
clone a voice you own, or one you have explicit, documented permission to
reproduce.** Cloning someone's voice without consent can violate right-of-publicity,
privacy, impersonation, and AI/deepfake-disclosure laws, which vary by
jurisdiction. You are solely responsible for how you use it and for complying with
applicable law. The authors provide this software as-is, with no warranty (see
LICENSE).

## License

This project's own code is licensed under the **Apache License 2.0** — see
[`LICENSE`](LICENSE).

It does **not** include or redistribute its dependencies or any machine-learning
model weights; those are fetched from PyPI and model hubs when you install and run
it, and each is covered by **its own license**. See [`NOTICE`](NOTICE) for a
summary of the notable third-party and model licenses. You are responsible for
complying with those terms — in particular, **check the model and LLM-provider
terms before any commercial use**.
