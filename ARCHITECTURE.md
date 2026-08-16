# Local Pipecat Voice Agent — Architecture

Personal voice assistant running entirely on a MacBook Pro (Apple Silicon, 18 GB unified
memory), cloned from the owner's voice, single user. Local mic/speaker via Pipecat
`LocalAudioTransport`. Hosted LLM; everything else local.

Written 2026-07-22. Facts about library state were verified against upstream repos/docs on
that date, then against the actually-installed packages (see "Verified against installed
versions" below), which resolved every `API-CHECK` and replaced budget guesses with
measured numbers where marked **[measured]**.

## Stack (final)

| Stage | Choice | Why / notes |
|---|---|---|
| VAD | Silero (Pipecat built-in) | `confidence=0.7, start_secs=0.2, stop_secs=0.2, min_volume=0.6` |
| Turn detection | Smart Turn v3 (`LocalSmartTurnAnalyzerV3`, built-in) | ~8M params, 12–30 ms CPU. Lets Silero `stop_secs` drop 0.8 → 0.2 s — the cheapest 600 ms available |
| STT | Parakeet TDT 0.6B **v3** via **parakeet-mlx** | Replaces parakeet-rs/ONNX plan: native Metal via MLX, pure Python, streaming API. v3 over v2: English WER delta <1 pt (noise in agent context), Spanish for free |
| LLM | Hosted, OpenAI-compatible (DeepSeek) | `OpenAILLMService(base_url=...)`. TTFT is the dominant, highest-variance latency term — measure from local network before committing |
| TTS | **Chatterbox-Turbo** (350M, MIT, official MPS) | Replaces base Chatterbox (500M): smaller, faster, built for agent loops, same `audio_prompt_path` cloning. **Batch-only upstream — no streaming synthesis** |
| Orchestration | Pipecat (current) | No first-class local Parakeet or local Chatterbox services → two custom services (this repo) |

Fallbacks: Chatterbox-Nano (110M, CPU) if MPS synth latency disappoints; low-TTFT LLM
provider (Groq/Cerebras-class) if DeepSeek p95 TTFT > 1 s.

## Pipeline graph

```
transport.input()          LocalAudioTransport: mic 16 kHz → InputAudioRawFrame
                           Silero VAD → UserStarted/StoppedSpeaking
                           Smart Turn v3 gates end-of-turn
                           speech during playback → StartInterruptionFrame (out-of-band)
stt                        ParakeetMLXSTTService (custom, SegmentedSTTService)
                           whole VAD segment → TranscriptionFrame
context_aggregator.user()  appends user msg, triggers LLM
llm                        OpenAILLMService → LLMTextFrame token stream
tts                        ChatterboxTTSService (custom, TTSService)
                           text aggregator: ClauseEagerAggregator (custom)
                           per chunk: blocking MPS generate() in executor, epoch-gated
                           → TTSAudioRawFrame (24 kHz, 40 ms slices) + TTSTextFrame
transport.output()         plays audio; forwards TTSTextFrame only as audio plays
context_aggregator.assistant()   AFTER output() → history records only *played* text
                                 (this is the barge-in state-reconciliation mechanism)
```

## Latency budget (end-of-user-speech → first audio out)

| Stage | Budget | Notes |
|---|---|---|
| VAD trailing silence | 200 ms | only viable with Smart Turn gating |
| Smart Turn inference | 15–30 ms | CPU |
| STT finalize (segmented) | ~250 ms **[measured]** | 244–256 ms warm for 4.9 s audio, perfect transcript (`scripts/check_stt.py`) |
| LLM RTT + TTFT | ~1050 ms **[measured]** | deepseek-v4-flash, thinking disabled, measured from a residential network: first content token p50 ≈ 1.0–1.15 s and it's baseline RTT/queue, not reasoning (first *reasoning* token also took ~1.1 s). Thinking-on adds ~350 ms more — disabled via `extra_body: {"thinking": {"type": "disabled"}}`. Your own numbers depend heavily on your location relative to the provider — measure before committing (`scripts/measure_ttft.py`) |
| LLM first-clause tokens | 100–250 ms | ~15–25 tokens |
| TTS first-chunk synth | ~1000–1500 ms **[measured]** | steady-state MPS RTF 0.73–0.97, so a chunk costs ≈ its own audio duration; first clause held to ~1 s of speech (`min_first_chars=24`) |
| Output buffering | 30–50 ms | |
| **Total** | **~2.6–3.0 s typical [measured components]** | both dominant terms measured worse than estimated: DeepSeek TTFT ~1.05 s (network-bound) + Turbo MPS RTF ~0.9. Sub-1.5 s needs *both* paths below |

Paths back toward the target, in order of expected payoff:
1. **TTS first-chunk**: the chatterbox-streaming fork (incremental s3gen decode →
   ~300–500 ms first audio) — evaluate Turbo compatibility; else benchmark
   Chatterbox-Nano (110M) or check for an mlx-audio Chatterbox port.
2. **LLM TTFT**: measured 1.0–1.1 s is route/queue-bound (first reasoning token also
   ~1.1 s), so no DeepSeek-side tuning will fix it. A geographically closer, low-TTFT
   provider (Groq/Cerebras-class) would land ~300–500 ms. DeepSeek is the chosen
   primary here; a fallback/fast-path provider is the lever if the total feels sluggish.
   Thinking is disabled either way (~350 ms saved, already wired).

Critical path is fully serial. After the first chunk, everything overlaps (chunk n plays
while n+1 synthesizes and the LLM streams). Biggest single win: LLM provider TTFT.
Second: eager first-clause TTS chunking (in the design). Later optimization: streaming
STT during speech cuts finalize to ~50 ms.

## Streaming handoffs

- Chunk policy (`ClauseEagerAggregator`): **ramped**. Every chunk flushes at a clause
  boundary (comma/semicolon/colon/dash) once past a minimum, and that minimum grows in
  proportion to the chunk just emitted (`min_chunk_chars=24`, `ramp=1.25`, capped at 180).
  The first chunk's floor stays at 24 chars, because it sets time-to-first-audio.

  This replaces the original "eager first chunk, then full sentences" policy, which was
  wrong: playback stays continuous only while `synth(chunk k) ≤ duration(chunk k-1)`, and
  at RTF ~0.9 that allows only ~1.1× growth per chunk. Emitting full sentences after a ~1s
  opener blew that budget badly — measured (`scripts/check_chunking.py`) at 4.94s of total
  dead air per reply, worst case 7.72s. Ramping cuts that to 2.20s / 3.08s at unchanged
  time-to-first-audio.

  Note the counter-intuitive result: **lengthening the first chunk is not the fix.** Going
  24 → 48 chars improves gaps by 0.06s while adding 0.37s before every first word. The
  bound on *subsequent* chunks is what buys continuity. Residual gaps are all long
  clause-free sentences, which have no legal split point — only streaming synthesis fixes
  those.
- Chatterbox upstream is batch-per-`generate()` (verified against repo). The
  davidbrowne17/chatterbox-streaming fork adds token-loop streaming for *base*
  Chatterbox; Turbo compatibility unknown. Evaluate only after the batch pipeline works
  — would cut first-audio by 300–500 ms.

## Barge-in design

Pipecat machinery (via `allow_interruptions=True`): input transport pushes
`StartInterruptionFrame` as an out-of-band system frame (bypasses processor queues);
LLM service cancels its HTTP stream (server-side abort best-effort); base TTSService
clears queued text; output transport flushes buffers.

What Pipecat can't do — the four real hazards:

1. **Uncancellable MPS kernel.** A `generate()` in flight runs to completion. Epoch
   counter in the TTS service: interruption bumps epoch (seen immediately in
   `process_frame`), orphaned job finishes into the void, output gated by epoch.
   **Never join/await the running synth on the interruption path** — blocking the event
   loop for a synth duration is the deadlock trap. Cost: one wasted GPU job briefly
   contending with the interrupting utterance's STT.
2. **Ghost audio.** Emit 40 ms audio slices (transport can only flush what isn't
   committed). OS/PortAudio buffer (~30–50 ms) is unflushable but inaudible. Hearing
   >100 ms ghost audio ⇒ some processor yielded one giant frame.
3. **Context reconciliation.** Structural: assistant aggregator after
   `transport.output()` records only played text (granularity: one TTS chunk). Never
   aggregate assistant text from upstream `LLMTextFrame`s.
4. **Straggler races.** Implement `handle_interruption`/`reset` on the text aggregator;
   don't cache state across interruptions in custom code.

False barge-in (echo — `LocalAudioTransport` has no AEC): wear a headset (the right
answer for single-user), or macOS Voice Isolation at runtime (never during reference
capture), or raise VAD confidence during agent playback. Don't build AEC.

## Turn detection tuning

With a turn analyzer, Silero `stop_secs=0.2` is only the trigger to *ask* smart-turn;
if incomplete, Pipecat waits up to `SmartTurnParams.stop_secs` (2.5 s) before forcing
end-of-turn. Snappy on clean completions, graceful on thinking pauses.

Tuning loop: log `(t_vad_stop, t_turn_decision, decision, t_transcript)` per turn.
Cut off mid-thought → raise smart-turn stop_secs. Sluggish → check smart-turn
probability logs. Triggers on breath/keyboard → confidence 0.8, raise min_volume.
Clips first syllable → start_secs 0.15.

## Persona layer

The system prompt (see `persona.py`) is the user-authored one with these deltas:
- Barge-in rule: history shows only what was actually said; never resume a cut-off
  thought unless asked.
- Spoken-form outputs: numbers/units/acronyms speakable ("eighteen gigs", "A-W-S").
- Disfluency cap promoted into the prompt: hedges/corrections in ~1 of 3–4 responses.
- Samples sanitized to work-appropriate register.
- Meta sections (corpus extension, tuning notes) kept out of the runtime prompt.

Few-shot vs distilled: hybrid, weighted distilled. 6–10 short samples (~600–900 tokens)
covering distinct modes (asking, pushing back, casual, teaching, disagreeing-with-agent)
beats 20 of one mode. DeepSeek prefix caching makes the stable prompt near-free after
turn one.

## Context management

Stable-prefix rolling window with lazy background summarization:
- System prompt + samples never change byte-for-byte → prefix cache eats the cost.
- Verbatim history to ~8k tokens.
- Then summarize oldest half into one system-role summary after the persona block; keep
  last ~20 turns verbatim. Summarize fire-and-forget during playback/idle, splice
  between turns (one cache-miss turn per splice — acceptable).
- Summary preserves commitments and open threads, not a transcript digest.

## Voice clone reference capture

- Record 60–90 s, harvest best 10–12 s contiguous window (Chatterbox uses ~10 s).
- In-product: the TUI's Clone screen records once, auto-slices spread candidate
  windows (`clone_voice.py`), synthesizes one fixed preview sentence per
  candidate with a single Chatterbox load, plays them back for A/B, and copies
  the winner to `voice_ref/reference.wav`. Preview model is unloaded on exit so
  the session doesn't hold two Chatterbox copies.
- 48 kHz WAV, **no processing**: disable Voice Isolation/AGC/noise suppression for
  capture. Quiet damped room; reverb in reference = permanent reverb in clone.
- Natural conversational speech, normal energy, include a question + a statement.
  Don't read, don't perform.
- Objective eval (`scripts/eval_clone.py`), optional: point it at
  `voice_ref/candidates/cand_*.wav`; scores each candidate on (1) speaker
  similarity — ECAPA/resemblyzer cosine vs held-out real recordings, calibrated
  against real-vs-real self-similarity (~0.80–0.90); (2) intelligibility —
  Parakeet WER of TTS output vs known text, target <5%; (3) ear A/B. Pick
  winner of (1) not failing (2).

## Resource budget (18 GB unified)

| Component | Framework | Resident | Pattern |
|---|---|---|---|
| Parakeet 0.6B fp16 | MLX | ~1.3–1.5 GB | 100–300 ms bursts after user speech |
| Chatterbox-Turbo + vocoder | PyTorch MPS | ~2–3 GB | 0.4–1 s bursts during agent speech |
| Smart Turn v3 | CPU | ~50 MB | per turn boundary |
| Silero | CPU | ~2 MB | continuous |

Total ~5–6 GB incl. runtime. Both stay resident on GPU; run nothing on CPU. Compute is
naturally time-multiplexed (STT while user speaks, TTS while agent speaks); only
overlap is the barge-in instant. Watch **thermals**: TTS RTF creeping up over 30+ min
sessions is throttling, not a leak — log per-chunk RTF over session time.

## Failure modes / observability

- LLM timeout: watchdog — no first token in ~3 s → cancel, retry once or play a
  pre-synthesized canned interstitial WAV. Handle ErrorFrames at task level.
- Audio device change (AirPods): PortAudio streams don't follow default-device
  switches. Catch stream errors, rebuild transport (models stay loaded, ~1 s recovery).
  Detect silent failure via "no VAD activity for N min on a healthy stream".
- Model load: load + warm both at startup, fail fast before opening the mic.
- VAD misfires: log Silero confidence histogram continuously.
- Metrics: **implemented** in `turn_metrics.py`. `TurnMetricsObserver` is a pipecat
  `BaseObserver` (registered via `PipelineWorker(observers=[...])`, so it watches every
  frame without sitting in the pipeline) that stitches `vad_stop → transcript →
  llm_first_token → tts_first_audio` into one INFO line per turn, plus a JSON record in
  `logs/turns-<timestamp>.jsonl`. Every latency regression is a delta in exactly one of
  those gaps.

  Two things this had to add beyond pipecat's built-ins:

  1. `can_generate_metrics()` returns `False` on the `FrameProcessor` base, and only
     vendor services override it — so the `start/stop_*_metrics()` calls already present
     in `stt_parakeet.py` and `tts_chatterbox.py` were silent no-ops. Both services now
     override it.
  2. Pipecat measures TTFB but has no notion of **playback gaps** — dead air *after*
     speech starts, when a chunk's synthesis outlasts the previous chunk's duration.
     That's the metric that governs whether a reply sounds fluent, so `TTSChunkLog`
     derives it directly from per-chunk synth timings (playback is modelled as: a chunk
     starts at `max(t_synth_done, previous_playback_end)`; any daylight is a gap).

  Per-chunk RTF is recorded alongside, which is the thermal-throttling signal — RTF
  drifting up over a long session is throttling, not a leak.

## Open questions — resolved

1. **Pipecat integration:** no first-class local Parakeet/Chatterbox services (Chatterbox
   only via hosted NVIDIA NIM). Custom services in this repo, subclassing
   `SegmentedSTTService`/`TTSService` for free interruption/metrics/aggregation.
2. **Parakeet v2 vs v3:** v3. WER delta is noise in agent context; Spanish free.
   Neither handles mid-utterance code-switching well.
3. **parakeet-rs:** skip. CoreML EP instability is real; MLX makes the problem
   disappear with no FFI bridge.
4. **Chatterbox variants:** Turbo. "Sub-200 ms" is Resemble's hosted service — do not
   budget from it. Early Turbo had MPS float64 dtype crashes (community-fixed) —
   verify installed version synthesizes on MPS before building on it. Check
   periodically whether mlx-audio has grown a Chatterbox port (would beat PyTorch-MPS).
   Rejected: Kokoro (no cloning), F5-TTS (non-commercial weights), Zonos (slower,
   shaky MPS), OpenAudio S1-mini (non-commercial).

## Verified against installed versions (2026-07-22)

Pinned: pipecat-ai 1.6.0, parakeet-mlx 0.5.2, chatterbox-tts 0.1.7, Python 3.12.
Pipecat 1.6 restructured the API vs. the draft above — the code reflects reality:

- **VAD + turn detection moved off the transport** into the user context aggregator:
  `LLMUserAggregatorParams(vad_analyzer=..., user_turn_strategies=...)`. The aggregator
  broadcasts `VADUserStartedSpeakingFrame`/`VADUserStoppedSpeakingFrame` **upstream** as
  well as downstream, which is what lets `SegmentedSTTService` (upstream of the
  aggregator) segment correctly. Smart Turn v3.2 ships bundled with pipecat (ONNX, no
  download) and is the *default* stop strategy.
- **Universal context**: `LLMContext` + `LLMContextAggregatorPair` replace the old
  OpenAI-specific context. `StartInterruptionFrame` → `InterruptionFrame`.
  Interruptions are on by default; `PipelineParams.allow_interruptions` is gone.
- **TTSService** now provides `_stream_audio_frames_from_iterator()` (chunking,
  resampling, context_id tagging) and cancels the `run_tts` consumer task on
  interruption — the epoch-counter design in the draft collapsed to "run blocking synth
  via `run_in_executor` and let cancellation abandon it". Custom text aggregators are
  installed by assigning `self._text_aggregator` post-init (no constructor kwarg in 1.6).
- **Dependency landmines** (all handled in pyproject/lock): pyaudio needs
  `brew install portaudio`; pkuseg needs numpy at build time
  (`tool.uv.extra-build-dependencies`); chatterbox pins `numpy<2` vs parakeet-mlx's
  `numpy>=2.2.5` — overridden via `tool.uv.override-dependencies`, validated by running
  both models; resemble-perth still imports `pkg_resources`, so setuptools is pinned
  `<81`.
- **Measured**: Parakeet warm finalize 244–256 ms for 4.9 s audio with a perfect
  transcript; Chatterbox-Turbo loads and synthesizes on MPS (no dtype crash in 0.1.7),
  steady-state RTF 0.73–0.97, ~20 s cold first-generate (shader compile — hence the
  warmup in service start()).

## Deferred: app-audio integration (Zoom etc.)

Requested, parked as the hardest feature. The plan when it happens: install
BlackHole (free virtual audio driver). Zoom's mic is set to "BlackHole 2ch" and the
agent's `AUDIO_OUT_DEVICE` points at the same BlackHole device, so the agent speaks
*into* the meeting; Zoom's speaker output goes to a Multi-Output/aggregate device that
includes a second BlackHole channel the agent's `AUDIO_IN_DEVICE` reads from, so the
agent hears the meeting. The env-var device routing and
`scripts/list_audio_devices.py` already exist as the foundation. Open problems: the
agent hears *everyone* in the meeting (needs speaker diarization or push-to-talk
gating to know when it's being addressed), and barge-in semantics in a multi-party
call need rethinking (probably half-duplex + explicit wake phrase).

## Build order (each stage gated)

1. **Audio skeleton** — LocalAudioTransport echo pipeline. Gate: mic+speaker work,
   AirPods toggle behavior known.
2. **STT isolated** — CLI on recorded file, then service logging transcripts.
   Gate: finalize <300 ms warm, accurate on owner's voice/vocabulary.
3. **LLM leg** — `scripts/measure_ttft.py`, 20 iterations against DeepSeek from the
   real network. Gate: p50/p95 known; p95 >1 s ⇒ decide on fallback provider now.
4. **Clone quality offline** — capture + eval loop, entirely outside Pipecat.
   Gate: chosen reference WAV with trusted similarity score, confirmed MPS synth + RTF.
5. **Full pipeline, no barge-in** — metrics on, stitched latency line.
   Gate: p50 <1.5 s over 20 real turns.
6. **Barge-in torture** — interrupt during first-chunk synth, mid-playback,
   rapid double interrupts. Gate: no ghost audio, no stalls, history = spoken text
   only, no stale-epoch frames.
7. **Persona + context management last.**
