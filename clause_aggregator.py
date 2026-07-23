"""Eager-first-clause text aggregation for the TTS stage.

The first chunk of an LLM response sets time-to-first-audio, so it is cut at the
first clause boundary (comma/semicolon/colon/dash) once a minimum length is
reached, instead of waiting for a full sentence. Later chunks are full sentences:
once audio is playing, synthesis time is hidden by playback and full sentences
buy better prosody for free. Chatterbox prosody degrades on tiny fragments,
hence the minimum-length gate before a clause flush.
"""

from collections.abc import AsyncIterator

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

CLAUSE_PUNCTUATION = ",;:—–"


class ClauseEagerAggregator(SimpleTextAggregator):
    # Default tuned to measured Chatterbox-Turbo MPS RTF (~0.9): the first chunk
    # costs roughly its own audio duration in synth time, so keep it near one
    # second of speech (~25 chars). Raise if first chunks sound clipped.
    def __init__(self, *, min_first_chars: int = 24, **kwargs):
        super().__init__(**kwargs)
        self._min_first_chars = min_first_chars
        self._first_flushed = False
        self._pending_clause = False

    async def _check_sentence_with_lookahead(self, char: str) -> Aggregation | None:
        result = await super()._check_sentence_with_lookahead(char)
        if result:
            self._first_flushed = True
            self._pending_clause = False
            return result
        if self._first_flushed:
            return None
        # Eager path, armed only until the first flush of a response. A clause
        # boundary needs whitespace lookahead: "1,000" or "10:30" must not
        # split, so the punctuation only counts when the next char is a space.
        if self._pending_clause:
            self._pending_clause = False
            if char.isspace():
                cut = len(self._text) - 1  # exclude the lookahead whitespace
                chunk = self._text[:cut]
                self._text = self._text[cut:]
                self._first_flushed = True
                return Aggregation(text=chunk.strip(), type=AggregationType.SENTENCE)
            return None
        if char in CLAUSE_PUNCTUATION and len(self._text) >= self._min_first_chars:
            self._pending_clause = True
        return None

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        async for aggregation in super().aggregate(text):
            yield aggregation

    async def flush(self) -> Aggregation | None:
        # End of LLM response: re-arm eager mode for the next response.
        self._first_flushed = False
        self._pending_clause = False
        return await super().flush()

    async def handle_interruption(self):
        self._first_flushed = False
        self._pending_clause = False
        await super().handle_interruption()

    async def reset(self):
        self._first_flushed = False
        self._pending_clause = False
        await super().reset()
