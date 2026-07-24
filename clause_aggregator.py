"""Ramped clause-boundary text aggregation for the TTS stage.

Two things set how responsive the agent feels, and they pull in opposite
directions:

  * **Time to first audio** — set by how much text the first chunk holds. Small
    first chunk, fast first word.
  * **Playback continuity** — set by whether each chunk lasts long enough to
    cover the synthesis of the next one.

Chatterbox-Turbo on MPS runs at RTF ~0.9, so a chunk costs roughly its own
audio duration to synthesize. Playback therefore stays gapless only while

    duration(chunk k) <= duration(chunk k-1) / RTF   (~1.1x at RTF 0.9)

The previous policy flushed the first chunk eagerly at a clause boundary and
then emitted **full sentences**, which breaks that inequality badly: a ~1s first
chunk followed by a 4s sentence needs ~3.6s of synthesis against 1s of cover,
so the speaker runs dry for ~2.6s mid-reply.

This aggregator instead ramps. It keeps splitting at clause boundaries for as
long as it usefully can, raising the minimum chunk size each time in proportion
to what was actually just emitted — so chunks grow at a rate playback can
absorb, and the ramp naturally stops mattering once the threshold exceeds
typical sentence length. Splits only ever land on clause punctuation; a chunk
below the floor is never emitted, because Chatterbox prosody degrades badly on
slivers.

Note that the ramp, not a bigger first chunk, is what buys continuity. Measured
over sample replies with `scripts/check_chunking.py` (RTF 0.9): the ramp alone
cuts total dead air 4.94s → 2.20s and the worst gap 7.72s → 3.08s at unchanged
time-to-first-audio. Doubling the first chunk to 48 chars on top of that buys a
further 0.06s of gap while costing 0.37s before every first word — which is why
the first-chunk floor stays small.

What the ramp cannot fix is a long sentence with no internal clause punctuation
(the residual ~3s worst case); splitting those needs mid-clause breaks that
Chatterbox renders badly. The real fix there is streaming synthesis — see
ARCHITECTURE.md, "Paths back toward the target".

Tunable at runtime for A/B-ing against the per-turn numbers in
logs/turns-*.jsonl (see turn_metrics.py) without editing code:

    PARROTBOX_MIN_FIRST_CHARS   first-chunk floor, default 24
    PARROTBOX_MIN_CHUNK_CHARS   prosody floor for later chunks, default 24
    PARROTBOX_CHUNK_RAMP        growth factor per chunk, default 1.25
    PARROTBOX_MAX_CHUNK_CHARS   stop growing past this, default 180
"""

import os
from collections.abc import AsyncIterator

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

CLAUSE_PUNCTUATION = ",;:—–"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# ~24 chars is about 1s of speech, and the smallest chunk Chatterbox renders
# with decent prosody. Kept small deliberately: it sets time-to-first-audio, and
# the ramp below — not a longer opening chunk — is what keeps playback gapless.
DEFAULT_MIN_FIRST_CHARS = int(_env_float("PARROTBOX_MIN_FIRST_CHARS", 24))
# The absolute prosody floor, applied to every chunk *after* the first. It must
# stay low and independent of the first-chunk target: a reply opening with a
# short sentence ("No.") buys almost no playback cover, so the chunk after it
# has to split at the earliest clause boundary available. Flooring later chunks
# at the larger first-chunk value re-creates exactly the gap this ramp exists to
# remove — measured at ~5s on a 136-char sentence following "No."
DEFAULT_MIN_CHUNK_CHARS = int(_env_float("PARROTBOX_MIN_CHUNK_CHARS", 24))
# Theory says ~1.11 at RTF 0.9. Emitted chunks overshoot the threshold (the split
# lands on the next clause boundary, not exactly at it), so the effective growth
# is larger than this number — hence a conservative default.
DEFAULT_RAMP = _env_float("PARROTBOX_CHUNK_RAMP", 1.25)
# Past this, eager splitting stops earning anything: sentences are usually
# shorter, so the natural sentence boundary wins anyway.
DEFAULT_MAX_CHUNK_CHARS = int(_env_float("PARROTBOX_MAX_CHUNK_CHARS", 180))


class ClauseEagerAggregator(SimpleTextAggregator):
    def __init__(
        self,
        *,
        min_first_chars: int = DEFAULT_MIN_FIRST_CHARS,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
        ramp: float = DEFAULT_RAMP,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._min_first_chars = min_first_chars
        self._min_chunk_chars = min_chunk_chars
        self._ramp = ramp
        self._max_chunk_chars = max_chunk_chars
        self._threshold = min_first_chars
        self._pending_clause = False

    def _advance(self, emitted_len: int) -> None:
        """Size the next chunk's floor against the chunk just emitted.

        Proportional to what was actually emitted rather than a fixed ramp from
        the previous threshold: a short chunk buys little playback cover, so the
        next one must stay small too. A fixed ramp would let the schedule run
        away after one short sentence.
        """
        grown = int(emitted_len * self._ramp)
        self._threshold = min(max(self._min_chunk_chars, grown), self._max_chunk_chars)

    def _rearm(self) -> None:
        self._threshold = self._min_first_chars
        self._pending_clause = False

    async def _check_sentence_with_lookahead(self, char: str) -> Aggregation | None:
        result = await super()._check_sentence_with_lookahead(char)
        if result:
            self._pending_clause = False
            self._advance(len(result.text))
            return result
        # Clause boundaries need whitespace lookahead: "1,000" and "10:30" must
        # not split, so the punctuation only counts when a space follows.
        if self._pending_clause:
            self._pending_clause = False
            if char.isspace():
                cut = len(self._text) - 1  # exclude the lookahead whitespace
                chunk = self._text[:cut].strip()
                self._text = self._text[cut:]
                self._advance(len(chunk))
                return Aggregation(text=chunk, type=AggregationType.SENTENCE)
            return None
        if char in CLAUSE_PUNCTUATION and len(self._text) >= self._threshold:
            self._pending_clause = True
        return None

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        async for aggregation in super().aggregate(text):
            yield aggregation

    async def flush(self) -> Aggregation | None:
        # End of LLM response: re-arm the ramp for the next one.
        self._rearm()
        return await super().flush()

    async def handle_interruption(self):
        self._rearm()
        await super().handle_interruption()

    async def reset(self):
        self._rearm()
        await super().reset()
