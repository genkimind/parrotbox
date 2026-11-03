# Usage guide

## Persona

Parrotbox loads `persona.local.md` when it exists and otherwise uses the
packaged fictional example in `src/parrotbox/persona.example.md`:

```bash
cp src/parrotbox/persona.example.md persona.local.md
```

Keep the persona concrete and behavioral. Six to ten short samples of real
speech usually guide style better than a long biography. The scenario entered
at startup is appended to the stable persona prompt so work meetings, casual
conversation, and interviews can change register without changing identity.

## Voice reference

The default TUI includes a **Clone a voice…** workflow:

1. Record at least 30 seconds in a quiet, damped room with normal speaking energy.
2. Let the app build several 12-second candidate windows and synthesize previews.
3. Listen to each preview and select the best candidate.

The winner becomes `voice_ref/reference.wav`. The directory is gitignored, and
the current session can use the new reference without restarting.

Record with macOS mic mode set to Standard. Voice Isolation, room reverb, and
aggressive processing can become part of the cloned voice. Include both a
question and a statement, and speak naturally rather than performing.

Only clone your own voice or one you have explicit permission to reproduce.

## Run modes

The Textual interface is the default:

```bash
uv run parrotbox
uv run parrotbox daily standup
```

The plain console flow is also available:

```bash
uv run parrotbox --console
uv run parrotbox --console daily standup
```

The console asks for a scenario, starting speed, mic mode, and response mode.
The first model start may take about a minute because weights load and MPS
shaders compile.

## TUI controls

| Key | Action |
|---|---|
| Up / Down | Select a voice slider |
| Left / Right | Adjust by 0.05; hold Shift for 0.50 |
| `t` | Type text for the cloned voice to speak verbatim |
| `m` | Mute or unmute the microphone |
| `c`, then `c` | Clear the conversation and model context |
| `r` | Respond to accumulated speech in meeting mode |
| `q` | Quit |

The live sliders control exaggeration, temperature, CFG weight, playback
speed, and volume. Changes are read for every synthesis chunk, so they take
effect during a reply without reloading the model.

Typed speech bypasses the LLM but is added to conversation context. Clearing
requires two presses because it permanently discards the current in-memory
conversation.

## Meeting mode

Meeting mode transcribes and accumulates turns without speaking. Press `r` to
request one response to everything heard since the previous response. An
utterance still in progress at the moment of the keypress belongs to the next
response.

Full duplex with headphones is the recommended meeting setup. In half duplex,
the microphone is muted while Parrotbox speaks, so other meeting audio during
that reply is lost.

## Echo and audio devices

Parrotbox has no acoustic echo cancellation. Use headphones for full-duplex
barge-in. With open speakers, select half duplex so the microphone does not
hear the cloned voice and interrupt it.

Route audio by device index when needed:

```bash
uv run python scripts/list_audio_devices.py
export AUDIO_IN_DEVICE=1
export AUDIO_OUT_DEVICE=3
```

## LLM configuration

| Variable | Purpose |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek key; defaults to `deepseek-flash` with thinking disabled |
| `LLM_PROVIDER` | Explicitly select `deepseek` or `custom` |
| `LLM_BASE_URL` | Custom OpenAI-compatible endpoint; auto-selects `custom` |
| `LLM_MODEL` | Override the profile model; required for `custom` |
| `LLM_API_KEY` | Generic auth override for any profile; optional for unauthenticated local servers |

DeepSeek is the default when no custom base URL is set. A custom endpoint uses
no provider-specific request options.

## Other configuration

| Variable | Purpose |
|---|---|
| `VOICE_REFERENCE` | Reference WAV path; defaults to `voice_ref/reference.wav` |
| `AUDIO_IN_DEVICE` | Input device index |
| `AUDIO_OUT_DEVICE` | Output device index |
| `PARROTBOX_MIN_FIRST_CHARS` | First TTS chunk floor |
| `PARROTBOX_MIN_CHUNK_CHARS` | Later TTS chunk floor |
| `PARROTBOX_CHUNK_RAMP` | Per-chunk growth factor |
| `PARROTBOX_MAX_CHUNK_CHARS` | Chunk growth ceiling |

For interpreting latency and tuning chunking, see
[PERFORMANCE.md](PERFORMANCE.md).
