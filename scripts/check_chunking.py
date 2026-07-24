"""Check and tune the TTS chunk policy — no models, runs in a second.

Two jobs:

  1. **Correctness** — the aggregator must not split inside "1,000" or "10:30",
     must lose no text, and must re-arm its ramp between responses.
  2. **Tuning** — simulates playback at the measured Chatterbox-Turbo MPS RTF so
     you can see time-to-first-audio and dead air for a given policy before
     committing to it in a live session.

Playback model: a chunk cannot start until it has been synthesized, and not
before the previous chunk finishes playing. Any daylight between those two is
silence the user hears. See clause_aggregator.py for why the ramp exists.

    uv run python scripts/check_chunking.py            # correctness + comparison
    PARROTBOX_MIN_FIRST_CHARS=24 uv run python scripts/check_chunking.py

Compare against the real numbers in logs/turns-*.jsonl once you've had a
session — gap_total_ms there is the ground truth this only estimates.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clause_aggregator import ClauseEagerAggregator  # noqa: E402

RTF = 0.9  # measured steady-state Chatterbox-Turbo on MPS (ARCHITECTURE.md)
CHARS_PER_SEC = 24.0  # conversational speech rate
UPSTREAM_SECS = 1.30  # measured STT finalize (~0.25) + LLM TTFT (~1.05)

SAMPLES = [
    "Yeah, I looked at that this morning, and honestly the numbers don't add up. "
    "We're seeing about thirty percent more traffic than last quarter, but the "
    "conversion rate dropped, so net revenue is basically flat.",
    "Sure. The build is failing because the lockfile drifted, so you'll want to "
    "re-run the sync before anything else.",
    "No. That's not going to work, because the API rate limits us at sixty "
    "requests a minute and we'd blow through that in the first ten seconds.",
    "Hmm, good question. I think the honest answer is that we don't really know "
    "yet, but my instinct says we should ship the smaller version first, get it "
    "in front of real users, and then decide whether the fancy version is worth "
    "the extra three weeks.",
]


async def split(text: str, **kwargs) -> list[str]:
    """Run text through the aggregator the way the LLM streams it."""
    agg = ClauseEagerAggregator(**kwargs)
    out: list[str] = []
    for i in range(0, len(text), 3):  # token-sized pieces
        async for aggregation in agg.aggregate(text[i : i + 3]):
            out.append(aggregation.text)
    tail = await agg.flush()
    if tail and tail.text.strip():
        out.append(tail.text)
    return out


def simulate(chunks: list[str]) -> tuple[float, list[float]]:
    """Return (time_to_first_audio, per-chunk gap seconds)."""
    now = 0.0
    playback_end: float | None = None
    gaps: list[float] = []
    ttfa = 0.0
    for chunk in chunks:
        audio = len(chunk) / CHARS_PER_SEC
        now += audio * RTF  # synthesis is serialized on one worker thread
        if playback_end is None:
            ttfa, gap, start = now, 0.0, now
        else:
            gap = max(0.0, now - playback_end)
            start = max(now, playback_end)
        gaps.append(gap)
        playback_end = start + audio
    return ttfa, gaps


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #


async def check_correctness() -> bool:
    failures: list[str] = []

    # A clause split must never land inside a number or a clock time. Enough
    # leading text that the threshold is already exceeded when they appear.
    text = (
        "So the short version is that we had about 1,000 users by 10:30 "
        "this morning, and it kept climbing all day."
    )
    chunks = await split(text)
    for c in chunks:
        stripped = c.rstrip()
        if stripped.endswith("1,") or stripped.endswith("10:"):
            failures.append(f"split inside a number/time: {c!r}")

    # No text may be lost or duplicated, under any policy.
    for sample in SAMPLES:
        for kwargs in ({}, {"min_first_chars": 24}, {"ramp": 2.0}):
            chunks = await split(sample, **kwargs)
            rejoined = "".join(chunks).replace(" ", "")
            if rejoined != sample.replace(" ", ""):
                failures.append(f"text lost with {kwargs} on {sample[:40]!r}")

    # The ramp must reset between responses: identical text in must give
    # identical chunks out, whether it is the first response or the third.
    agg = ClauseEagerAggregator()
    shapes = []
    for _ in range(3):
        out = []
        for ch in SAMPLES[0]:
            async for a in agg.aggregate(ch):
                out.append(a.text)
        tail = await agg.flush()
        if tail and tail.text.strip():
            out.append(tail.text)
        shapes.append([len(c) for c in out])
    if not all(s == shapes[0] for s in shapes):
        failures.append(f"ramp did not re-arm between responses: {shapes}")

    # Interruption must also re-arm, or the chunk after a barge-in inherits a
    # stale (too large) threshold and opens a gap.
    agg = ClauseEagerAggregator()
    async for _ in agg.aggregate("Something long enough to move the threshold up, and more."):
        pass
    await agg.handle_interruption()
    if agg._threshold != agg._min_first_chars:
        failures.append("handle_interruption did not reset the ramp")

    for f in failures:
        print(f"  FAIL  {f}")
    if not failures:
        print("  all correctness checks passed")
    return not failures


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #


async def compare() -> None:
    policies = [
        ("previous: 24 first, then full sentences",
         dict(min_first_chars=24, min_chunk_chars=24, ramp=999.0, max_chunk_chars=10**6)),
        ("ramp, 24 first", dict(min_first_chars=24)),
        ("ramp, 36 first", dict(min_first_chars=36)),
        ("ramp, 48 first", dict(min_first_chars=48)),
        ("ramp, 60 first", dict(min_first_chars=60)),
    ]
    print(f"\n  RTF {RTF}, {CHARS_PER_SEC:.0f} chars/sec, upstream {UPSTREAM_SECS:.2f}s\n")
    print(f"  {'policy':<42}{'first audio':>12}{'gaps':>9}{'worst':>8}")
    print("  " + "-" * 69)
    for label, kwargs in policies:
        ttfas, totals, worst = [], [], []
        for sample in SAMPLES:
            chunks = await split(sample, **kwargs)
            ttfa, gaps = simulate(chunks)
            ttfas.append(ttfa)
            totals.append(sum(gaps))
            worst.append(max(gaps) if gaps else 0.0)
        n = len(SAMPLES)
        print(
            f"  {label:<42}{sum(ttfas) / n + UPSTREAM_SECS:>11.2f}s"
            f"{sum(totals) / n:>8.2f}s{max(worst):>7.2f}s"
        )


async def show_current() -> None:
    print("\n  chunk shapes with the current settings:\n")
    for sample in SAMPLES:
        chunks = await split(sample)
        ttfa, gaps = simulate(chunks)
        print(f"  first audio {ttfa + UPSTREAM_SECS:.2f}s, gaps {sum(gaps):.2f}s")
        for chunk, gap in zip(chunks, gaps):
            flag = f"   <-- {gap:.2f}s of silence" if gap > 0.05 else ""
            print(f"    [{len(chunk):3d}ch {len(chunk) / CHARS_PER_SEC:4.1f}s] {chunk[:54]!r}{flag}")
        print()


async def main() -> int:
    agg = ClauseEagerAggregator()
    print(
        f"policy: min_first={agg._min_first_chars} min_chunk={agg._min_chunk_chars} "
        f"ramp={agg._ramp} max={agg._max_chunk_chars}\n"
    )
    ok = await check_correctness()
    await compare()
    await show_current()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
