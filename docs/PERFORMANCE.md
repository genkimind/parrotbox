# Performance and measurement

Parrotbox treats latency as a pipeline budget, not a single benchmark number.
The critical path to first audio is serial:

```text
end of speech → turn decision → STT finalize → LLM first clause → TTS first chunk → output
```

After playback starts, LLM generation, synthesis, and audio output overlap.
That creates a second user-visible metric: whether synthesis keeps up with
playback without inserting silence between chunks.

## Reference measurements

These values were recorded on the project's reference setup: an Apple Silicon
MacBook Pro with 18 GB unified memory, Python 3.12, Pipecat 1.6.0,
parakeet-mlx 0.5.2, and chatterbox-tts 0.1.7. Network results reflect the
author's route to DeepSeek and will vary by location and provider load.

| Measurement | Observed result | How it was measured |
|---|---:|---|
| Parakeet warm finalization | 244–256 ms | 4.9 s clean audio via `scripts/check_stt.py` |
| DeepSeek first content token | ~1.0–1.15 s | Streaming chat completion via `scripts/measure_ttft.py` |
| Chatterbox-Turbo warm RTF | 0.73–0.97 | MPS synthesis via `scripts/check_tts.py` |
| Typical first audio | ~2.6–3.0 s | Sum of measured dominant stages plus turn/output overhead |

RTF is real-time factor: `0.9` means one second of audio takes roughly 0.9
seconds to synthesize. Cold-start model loading and the first MPS shader compile
are excluded from warm figures; the application warms models before opening a
session.

## Reproduce the stage checks

```bash
# No models; validates text preservation and compares chunk policies
uv run python scripts/check_chunking.py

# Apple Silicon and model downloads required
uv run python scripts/check_stt.py
uv run python scripts/check_tts.py

# Uses the configured hosted/local endpoint; defaults to 20 requests
uv run python scripts/measure_ttft.py 20
```

Change one stage at a time. A faster LLM does not compensate for a TTS playback
gap, and larger first chunks may reduce gaps while making time-to-first-audio
worse.

## Runtime metrics

Each turn produces a readable log line and a JSONL record under `logs/`:

```text
TURN ttfa=2.35s (stt 0.25s | llm 1.04s | tts 1.05s) chunks=3 audio=6.00s gaps=1.60s max=1.60s
```

- `ttfa`: end-of-user-speech to first synthesized audio.
- `stt`, `llm`, `tts`: the critical-path components of `ttfa`.
- `gaps`: total silence after playback has begun because the next chunk was not ready.
- `max`: longest individual playback gap.
- per-chunk RTF in JSONL: useful for spotting thermal throttling during long sessions.

If `gaps` is near zero but the agent still feels slow, inspect the `ttfa`
breakdown. If `ttfa` is acceptable but speech stalls mid-reply, tune chunking or
evaluate streaming synthesis.

## Chunk policy

Chatterbox-Turbo is batch-per-chunk. `ClauseEagerAggregator` emits an early
first clause, then grows later chunks gradually so the previous chunk can cover
the synthesis time of the next one. It splits only at clause or sentence
boundaries to preserve prosody.

The policy can be A/B tested without editing code:

| Variable | Default | Effect |
|---|---:|---|
| `PARROTBOX_MIN_FIRST_CHARS` | 24 | First-chunk floor; larger delays the first word |
| `PARROTBOX_MIN_CHUNK_CHARS` | 24 | Prosody floor for later chunks |
| `PARROTBOX_CHUNK_RAMP` | 1.25 | Growth factor; lower is safer for continuity |
| `PARROTBOX_MAX_CHUNK_CHARS` | 180 | Ceiling for ramped chunk growth |

The remaining worst case is a long sentence with no internal clause boundary.
No punctuation-aware policy can split it safely; true streaming TTS is the
structural fix.

## Practical acceptance gates

The project uses these as engineering signals, not product SLAs:

- warm STT finalize below 300 ms on the reference clip;
- warm TTS RTF below 1.0;
- measured LLM p50 and p95 from the actual network before selecting a provider;
- zero text loss across chunk boundaries;
- no stale audio after interruption; and
- conversation history matching only audio that was actually played.

See [../ARCHITECTURE.md](../ARCHITECTURE.md) for the full concurrency,
interruption, and context-reconciliation design.
