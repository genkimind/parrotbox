"""Rule-based text normalization for the TTS stage.

Chatterbox speaks symbols literally ("..." becomes "dot dot dot"), so text is
rewritten into speakable form before synthesis. Runs as a Pipecat text filter,
per aggregated chunk, after ClauseEagerAggregator — which splits "..." across
chunks ("Well.." then "."), so the edge rules match 1-2 dots at chunk
boundaries. A chunk that normalizes to nothing is dropped by TTSService before
synthesis. The filtered text is also what lands in the assistant context, so
history records what was actually said.
"""

import re

from pipecat.utils.text.base_text_filter import BaseTextFilter

RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"…"), "..."),  # unify unicode ellipsis
    (re.compile(r"^\s*\.+\s*"), ""),  # leading dots = aggregator split artifact
    (re.compile(r"\.{2,}\s*$"), "."),  # trailing ellipsis -> sentence stop
    (re.compile(r"\s*\.{3,}\s*"), ", "),  # mid-sentence ellipsis -> pause
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),  # [text](url) -> text
    (re.compile(r"[*_`#]+"), ""),  # stray markdown emphasis/code/headers
    (re.compile(r"\s*[—–]\s*|\s+--?\s+"), ", "),  # dashes -> pause
    (re.compile(r"(?<=\d)\s*%"), " percent"),
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile(r"(?<=\d)\s*°"), " degrees"),
    (re.compile(r"~\s*(?=\d)"), "about "),
    # cleanup passes, always last:
    (re.compile(r"\s+([,.;:!?])"), r"\1"),  # no space before punctuation
    (re.compile(r"([,;:])\s*(?=[,.;:!?])"), ""),  # collapse doubled punctuation
    (re.compile(r"\s{2,}"), " "),
]


def normalize(text: str) -> str:
    for pattern, repl in RULES:
        text = pattern.sub(repl, text)
    return text.strip()


class TTSNormalizeFilter(BaseTextFilter):
    async def filter(self, text: str) -> str:
        return normalize(text)
