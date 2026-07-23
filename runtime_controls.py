"""Live-adjustable synthesis controls shared between the TUI and the TTS service.

Chatterbox exposes exaggeration/temperature/cfg_weight as per-generate() call
arguments, and speed is a post-synth time-stretch. Because synthesis runs once
per clause chunk, a single mutable instance read at the top of every _synth()
call lets the TUI move a slider and hear the change within a sentence — no
model reload, no pipeline restart.
"""

from dataclasses import dataclass, replace


@dataclass
class ParamSpec:
    """One adjustable slider: which field it drives and its bounds."""

    label: str
    attr: str
    lo: float
    hi: float
    step: float
    help: str
    emphasis: str = "normal"  # "thin" | "normal" | "thick" — bar thickness (not length)
    note: str = ""  # short inline hint shown after the value (e.g. "adjust this first")


# Order here is the order shown in the TUI.
PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec(
        "Exaggeration", "exaggeration", 0.0, 4.5, 0.05,
        "adjust this first — emotion/intensity; higher is more excited and animated",
        emphasis="thick",
        note="◀ adjust this first",
    ),
    ParamSpec(
        "Temperature", "temperature", 0.1, 4.3, 0.05,
        "delivery variability — higher is livelier, less predictable",
        emphasis="thin",
    ),
    ParamSpec(
        "CFG weight", "cfg_weight", 0.0, 3.0, 0.05,
        "pacing/adherence — lower is slower and more deliberate",
        emphasis="thin",
    ),
    ParamSpec(
        "Speed", "speed", 0.5, 5.0, 0.05,
        "post-synth playback rate — above 1.0 is faster but can sound echoey",
    ),
    ParamSpec(
        "Volume", "volume", 0.2, 1.5, 0.05,
        "output loudness — 1.0 is 100%; up to 1.5 overboosts +50% (louder, clips hot)",
    ),
)


@dataclass
class RuntimeControls:
    """Mutable synthesis parameters. Read live by ChatterboxTTSService._synth."""

    exaggeration: float = 0.5
    temperature: float = 0.8
    cfg_weight: float = 0.0
    speed: float = 1.0
    volume: float = 1.0

    def snapshot(self) -> "RuntimeControls":
        """A detached copy — safe to read without racing a concurrent adjust."""
        return replace(self)

    def adjust(self, spec: ParamSpec, steps: int) -> float:
        """Nudge one field by `steps` increments, clamped to the spec's bounds.

        Rounded to the step grid so values stay clean (0.55, not 0.5500001).
        Returns the new value.
        """
        current = getattr(self, spec.attr)
        raw = current + steps * spec.step
        clamped = max(spec.lo, min(spec.hi, raw))
        # Snap to the step grid relative to lo to avoid float drift.
        snapped = spec.lo + round((clamped - spec.lo) / spec.step) * spec.step
        snapped = max(spec.lo, min(spec.hi, round(snapped, 4)))
        setattr(self, spec.attr, snapped)
        return snapped
