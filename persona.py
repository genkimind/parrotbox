"""Loads the system prompt from persona.md — edit that file, not this one.

The scenario section is appended at the END so the persona block stays
byte-identical across sessions: DeepSeek's prefix cache keys on the stable
prefix, making the resent prompt near-free after turn one.
"""

from pathlib import Path

PERSONA_FILE = Path(__file__).resolve().parent / "persona.md"

SCENARIO_TEMPLATE = """

## Current situation

Right now, this specific conversation is happening in the following context: \
{scenario}

Let that shape register, length, and what you assume the other person knows. \
A work standup gets terse status-talk; a family conversation gets warmth and \
zero jargon; an interview gets fuller answers. Stay the same person either way \
— the situation changes the register, not the character."""

MEETING_TEMPLATE = """

## Meeting mode

You have been listening silently and were just asked to speak. The user \
messages above may span several utterances, possibly from different speakers \
and topics. Respond once, to what is directly relevant or asked of you — do \
not recap everything you heard, and do not answer points that already \
resolved themselves in the discussion."""


def build_system_prompt(scenario: str | None = None, meeting: bool = False) -> str:
    prompt = PERSONA_FILE.read_text(encoding="utf-8").strip()
    if scenario and scenario.strip():
        prompt += SCENARIO_TEMPLATE.format(scenario=scenario.strip())
    if meeting:
        prompt += MEETING_TEMPLATE
    return prompt
